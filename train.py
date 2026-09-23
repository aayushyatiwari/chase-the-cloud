import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import yaml
import wandb
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from src.dataset import Clouds
from src.manifest import split_indices
from src.models.convlstm import ConvLSTM
from src.models.simvp import SimVP
from src.models.simvp2 import SimVPv2
from src.models.predRNN import PredRNN
from src.models.phydnet import PhyDNet
from src.models.residual import ResidualWrapper
from src.engine import Trainer, EarlyStopping
from src.utils import latest_checkpoint, CombinedLoss
import time


def fmt_duration(seconds):
    """Seconds as h:mm:ss, so a long run is readable at a glance."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    """
    Make a run repeatable: the same seed gives the same starting weights, the
    same random crops and the same shuffle order. Without this every run differs
    slightly, so two architectures cannot be fairly compared -- the gap between
    them might just be luck.

    GPU kernels can still add tiny non-determinism. Setting
    torch.backends.cudnn.deterministic = True removes that too, but is slower.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def check_output_shape(model, loader, device):
    """
    Fail loudly when the model's output does not match the target.

    MSELoss broadcasts rather than raising, so a (B, 4, H, W) prediction scored
    against a (B, 1, H, W) target trains happily against nonsense. That is a
    real possibility here: the input is wider than the target by design, and
    only the model's head decides how wide the output is.
    """
    inputs, targets = next(iter(loader))
    model.eval()
    with torch.no_grad():
        out = model(inputs.to(device))
    model.train()
    if tuple(out.shape) != tuple(targets.shape):
        raise ValueError(
            f"Model outputs {tuple(out.shape)} but the target is {tuple(targets.shape)}. "
            f"MSELoss would broadcast these instead of erroring. Check "
            f"data.target_channels against the model's output width."
        )
    return inputs, targets, out


def run_dry(model, criterion, optimizer, train_loader, val_loader, device, steps):
    """
    A few real steps on real batches, then exit.

    Also prints the pixel/gradient split of each loss, which is how beta gets
    calibrated: on [0, 1] frames the two terms are not on the same scale, so a
    beta chosen blind can leave the gradient term contributing nothing.

    Checks the things that only fail once training is actually under way --
    shapes, dtypes, GPU memory, and how long an epoch will really take -- and
    writes no checkpoints and logs nothing, so it costs a minute instead of an
    afternoon.
    """
    print(f"\n=== DRY RUN: {steps} train steps, {steps} val steps ===")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    inputs, targets, out = check_output_shape(model, train_loader, device)
    print(f"inputs  {tuple(inputs.shape)} {inputs.dtype} "
          f"[{inputs.min():.4f}, {inputs.max():.4f}]")
    print(f"targets {tuple(targets.shape)} {targets.dtype} "
          f"[{targets.min():.4f}, {targets.max():.4f}]")
    print(f"outputs {tuple(out.shape)}  <- matches target")

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    t0 = time.time()
    done = 0
    for i, (inputs, targets) in enumerate(train_loader):
        if i >= steps:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        done += 1
        # The two terms are on different scales, so the split is what tells you
        # whether beta is large enough for the gradient term to matter at all.
        terms = getattr(criterion, 'last_terms', None)
        split = (f"  (pixel {terms['pixel']:.4f} + gdl {terms['gdl']:.4f})"
                 if terms else "")
        print(f"  train step {i + 1}/{steps}  loss {loss.item():.4f}{split}")
    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_s = (time.time() - t0) / max(done, 1)

    model.eval()
    t0 = time.time()
    done_val = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            if i >= steps:
                break
            loss = criterion(model(inputs.to(device)), targets.to(device))
            done_val += 1
            print(f"  val   step {i + 1}/{steps}  loss {loss.item():.4f}")
    if device.type == 'cuda':
        torch.cuda.synchronize()
    val_s = (time.time() - t0) / max(done_val, 1)

    print(f"\nPer step: train {train_s * 1000:.0f} ms, val {val_s * 1000:.0f} ms")
    print(f"Epoch estimate: train {fmt_duration(train_s * len(train_loader))} "
          f"({len(train_loader)} steps) + val {fmt_duration(val_s * len(val_loader))} "
          f"({len(val_loader)} steps) = "
          f"{fmt_duration(train_s * len(train_loader) + val_s * len(val_loader))}")
    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"Peak GPU memory: {peak:.2f}GB of {total:.1f}GB "
              f"({100 * peak / total:.0f}%)")
    print("=== DRY RUN OK -- nothing saved, nothing logged ===\n")


def build_model(config, dataset, device):
    """
    The model named by config.model.type, sized from the data.

    Every architecture takes all of the dataset's channels as input and emits only
    dataset.target_channels -- the input is wider than the target by design, and
    the predicted channels come first. Returns (model, arch_tag), where arch_tag
    goes into the run name so a checkpoint can never be mistaken for one from a
    different architecture.
    """
    m = config['model']
    model_type = m['type']
    if model_type == 'convlstm':
        model = ConvLSTM(
            input_dim=dataset.C,
            hidden_dim=m['hidden_dim'],
            kernel_size=m['kernel_size'],
            num_layers=m['num_layers'],
        )
        arch_tag = f"L{m['num_layers']}_h{m['hidden_dim']}"
    elif model_type == 'simvp':
        model = SimVP(
            shape_in=(config['data']['T'], dataset.C,
                      config['data']['crop_size'], config['data']['crop_size']),
            hid_S=m['hid_S'],
            hid_T=m['hid_T'],
            N_S=m['N_S'],
            N_T=m['N_T'],
            T_out=1,
            groups=m['groups'],
            out_channels=dataset.target_channels,
        )
        arch_tag = f"hidS{m['hid_S']}_NT{m['N_T']}"
    elif model_type == 'simvp2':
        # Same encoder/decoder as simvp, gSTA MetaFormer translator instead of
        # Inception. Keep hid_S/hid_T/N_S/N_T equal to the simvp run if the two
        # are meant to be compared -- then only the translator differs.
        model = SimVPv2(
            shape_in=(config['data']['T'], dataset.C,
                      config['data']['crop_size'], config['data']['crop_size']),
            hid_S=m['hid_S'],
            hid_T=m['hid_T'],
            N_S=m['N_S'],
            N_T=m['N_T'],
            T_out=1,
            mlp_ratio=float(m.get('mlp_ratio', 4.0)),
            drop=float(m.get('drop', 0.0)),
            spatio_kernel=m.get('spatio_kernel', 21),
            out_channels=dataset.target_channels,
        )
        arch_tag = f"hidS{m['hid_S']}_NT{m['N_T']}_gsta{m.get('spatio_kernel', 21)}"
    elif model_type == 'predrnn':
        model = PredRNN(
            input_dim=dataset.C,
            hidden_dim=m['hidden_dim'],
            kernel_size=m['kernel_size'],
            num_layers=m['num_layers'],
            out_channels=dataset.target_channels,
        )
        arch_tag = f"L{m['num_layers']}_h{m['hidden_dim']}"
    elif model_type == 'phydnet':
        model = PhyDNet(
            input_dim=dataset.C,
            out_channels=dataset.target_channels,
            nf=m.get('nf', 32),
            latent_dim=m.get('latent_dim', 64),
            phy_hidden_dims=m.get('phy_hidden_dims', 49),
            phy_layers=m.get('phy_layers', 1),
            phy_kernel_size=m.get('phy_kernel_size', 7),
            conv_hidden_dims=tuple(m.get('conv_hidden_dims', (128, 128, 64))),
            conv_layers=m.get('conv_layers', 3),
            conv_kernel_size=m.get('kernel_size', 3),
        )
        arch_tag = (f"lat{m.get('latent_dim', 64)}_phy{m.get('phy_layers', 1)}"
                    f"x{m.get('phy_kernel_size', 7)}")
        # The moment regulariser that makes PhyCell a PDE is a loss term, and
        # nothing adds it yet -- see the note at the top of src/models/phydnet.py.
        print("PhyDNet: moment regularisation is NOT applied; PhyCell's filters "
              "are unconstrained until model.moment_loss() is added to the loss.")
    else:
        raise ValueError(
            f"Unknown model.type: {model_type!r} "
            f"(expected 'convlstm', 'simvp', 'simvp2', 'predrnn' or 'phydnet')")

    # Neither mode bounds its output, so the two stay comparable.
    if m.get('residual'):
        model = ResidualWrapper(model, out_channels=dataset.target_channels)
        arch_tag += "_res"
        print("Residual mode: model predicts the change from the last input frame")

    return model.to(device), arch_tag


def build_criterion(config):
    """
    The loss named by config.train.loss, as

        L = alpha * pixel(pred, target) + beta * GDL(pred, target)

    where `pixel` is MSE or L1 and GDL scores the edges (see
    src.utils.gradient_difference). Setting a weight to zero drops that term:
    beta = 0 is plain pixel training, alpha = 0 is gradients alone -- which is
    blind to overall brightness, so it is an ablation rather than a candidate.

    Returns (criterion, loss_tag), where loss_tag goes into the run name. A
    checkpoint stores only its loss as a bare number, with nothing saying which
    objective produced it, so the name is the only thing keeping runs under
    different losses apart.
    """
    cfg = config['train'].get('loss') or {}
    pixel_type = cfg.get('pixel', 'mse')
    # float() guards against YAML reading e.g. 1.0e-2 as a string
    alpha = float(cfg.get('alpha', 1.0))
    beta = float(cfg.get('beta', 0.0))
    p = int(cfg.get('gdl_p', 1))

    pixel_losses = {'mse': nn.MSELoss, 'l1': nn.L1Loss}
    if pixel_type not in pixel_losses and pixel_type not in (None, 'none'):
        raise ValueError(
            f"Unknown train.loss.pixel: {pixel_type!r} (expected 'mse', 'l1' or 'none')")
    if pixel_type in (None, 'none'):
        alpha = 0.0
        pixel_type = 'mse'  # unused at alpha = 0, but keeps the module constructible
    if alpha == 0.0 and beta == 0.0:
        raise ValueError("train.loss has alpha = beta = 0; there is nothing to minimise.")

    criterion = CombinedLoss(pixel_losses[pixel_type](), alpha=alpha, beta=beta, p=p)

    parts = ([pixel_type] if alpha else []) + ([f"gdl{p}"] if beta else [])
    loss_tag = "+".join(parts)
    if beta:
        loss_tag += f"_a{alpha:g}b{beta:g}"
    print(f"Loss: {loss_tag}  (L = {alpha:g} * {pixel_type} + {beta:g} * GDL p={p})")
    return criterion, loss_tag


def main(args):
    # 1. Load Configuration
    config = load_config(args.config)
    set_seed(config['train']['seed'])

    # 2. Hardware Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Pin a process to one GPU with CUDA_VISIBLE_DEVICES, so two experiments can
    # share the node without either seeing the other's card.
    if device.type == 'cuda':
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)}, "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")
    else:
        print("Using device: cpu")

    # 3. Data Preparation
    # Split by time only -- train and val share geography on purpose, since the
    # model is deployed over this same sector. split_indices keeps every frame
    # of a window inside one split, so no frame is shared across the boundary.
    T = config['data']['T']
    crop_size = config['data']['crop_size']

    print("Splits:")
    idx = split_indices(config['data']['manifest_path'], config['data']['splits'])

    # One tiling for every split, so each pixel is trained on and scored equally.
    common = dict(manifest_path=config['data']['manifest_path'], T=T, crop_size=crop_size,
                  crop_stride=config['data']['crop_stride'],
                  lut_path=config['data']['lut_path'],
                  norm_ranges_path=config['data']['norm_ranges_path'],
                  target_channels=config['data'].get('target_channels', 1))

    train_dataset = Clouds(**common, window_range=idx['train'])
    val_dataset = Clouds(**common, window_range=idx['val'])
    grid = train_dataset.crops

    print(f"Frames are {train_dataset.C}x{train_dataset.H}x{train_dataset.W}, crop {crop_size}")
    print(f"Channels in {train_dataset.channels}, predicting the first "
          f"{train_dataset.target_channels}")
    print(f"Train: {len(train_dataset)} samples ({len(idx['train'])} windows x {len(grid)} crops)")
    print(f"Val:   {len(val_dataset)} samples ({len(idx['val'])} windows x {len(grid)} crops)")

    # Workers hand batches over through /dev/shm, which containers often cap at
    # 64MB. If a worker dies with a bus error, that is why -- drop num_workers
    # to 0, or restart the container with a larger --shm-size.
    loader_args = dict(batch_size=config['train']['batch_size'],
                       num_workers=config['train']['num_workers'])
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    
    # 4. Initialize Model, Optimizer, and Loss Function
    model, arch_tag = build_model(config, train_dataset, device)

    # float() guards against YAML parsing e.g. 3e-5 as a string
    lr = float(config['train']['lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion, loss_tag = build_criterion(config)
    # The objective is part of what a checkpoint is, so it goes in the name.
    # Plain MSE is the old default and keeps the old naming, so earlier runs
    # stay comparable at a glance.
    if loss_tag != 'mse':
        arch_tag += f"_{loss_tag}"

    # Cut the learning rate once validation stops improving. A fixed rate leaves
    # both losses flat well before early stopping fires -- the optimiser is
    # taking steps too big to settle into the minimum it is already sitting in.
    # lr_patience must stay below early_stopping_patience, or training ends
    # before the rate is ever reduced.
    scheduler = None
    if config['train'].get('lr_schedule'):
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(config['train'].get('lr_factor', 0.5)),
            patience=config['train'].get('lr_patience', 3),
            min_lr=float(config['train'].get('lr_min', 1e-6)),
        )
        print(f"LR schedule: halve after {scheduler.patience} epochs without improvement")

    if args.dry_run:
        run_dry(model, criterion, optimizer, train_loader, val_loader,
                device, args.dry_run)
        return

    # Cheap once-per-run guard against a silently broadcast loss.
    check_output_shape(model, train_loader, device)

    early = EarlyStopping(
        patience=config['train'].get('early_stopping_patience', 10),
        # float() guards against YAML reading 1.0e-5 as a string
        min_delta=float(config['train'].get('early_stopping_min_delta', 1e-5)),
    )
    best_val_loss = float('inf')
    best_ckpt = None

    # 5. Initialize the Trainer (The Engine)
    # e.g. 20260828_161422_convlstm_L3_h64 -- timestamp plus architecture, so a
    # stale checkpoint can never be mistaken for one matching the current config.
    run_name = (args.name
                or config['logging'].get('run_name')
                or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                   f"_{config['model']['type']}_{arch_tag}")
    print(f"Run name: {run_name}")

    # 2. Initialize wandb, now that the run has a name.
    # Named explicitly so two concurrent experiments are told apart at a glance
    # instead of getting wandb's random adjective-noun pairs. `group` ties
    # related runs together; `tags` filter them.
    if config['logging']['use_wandb'] and not args.dry_run:
        log_cfg = config['logging']
        wandb.init(
            project=log_cfg['project'],
            name=run_name,
            group=log_cfg.get('group') or None,
            tags=log_cfg.get('tags') or None,
            notes=log_cfg.get('notes') or None,
            config=config,  # log hyperparameters
        )

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        device,
        checkpoint_dir=config['train']['checkpoint_dir'],
        run_name=run_name
    )
    
    # 6. Optionally resume from a previous checkpoint
    # config: train.resume_from -- a path, or 'latest' for the newest checkpoint.
    # Epoch numbering and best_val_loss continue from the checkpoint so a resumed
    # run neither restarts the count nor saves a checkpoint worse than the one it
    # loaded. Checkpoints still go to a fresh run directory.
    start_epoch = 0
    resume_from = config['train'].get('resume_from')
    if resume_from:
        if resume_from == 'latest':
            resume_from = latest_checkpoint(config['train']['checkpoint_dir'], model=model)
        start_epoch, best_val_loss = trainer.load_checkpoint(resume_from, lr=lr)
        early.best_loss = best_val_loss

    # 7. The Main Training Loop
    epochs = config['train']['epochs']
    start = time.time()
    print(f"Starting Training... ({epochs} epochs max, "
          f"began {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

    epochs_run = 0
    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
        epoch_start = time.time()

        # Train
        avg_train_loss = trainer.train_one_epoch(train_loader, epoch)
        train_secs = time.time() - epoch_start

        # Validate
        val_start = time.time()
        val_metrics = trainer.validate(val_loader)
        val_secs = time.time() - val_start

        # Step the schedule on the same number early stopping watches, so the
        # rate is always cut a few epochs before the run is given up on.
        lr_before = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != lr_before:
            print(f"    LR reduced: {lr_before:.2e} -> {current_lr:.2e}")

        epoch_secs = time.time() - epoch_start
        epochs_run += 1
        # Guess the finish time from the average epoch so far, so a long run
        # can be left alone with some idea of when to come back.
        elapsed = time.time() - start
        remaining = (epochs - epochs_run) * (elapsed / epochs_run)

        print(f"==> Epoch {epoch} Complete.")
        print(f"    Train Loss: {avg_train_loss:.4f}")
        print(f"    Val Loss:   {val_metrics['loss']:.4f}  (persistence {val_metrics['persistence_loss']:.4f})")
        print(f"    Val SSIM:   {val_metrics['ssim']:.4f}  (persistence {val_metrics['persistence_ssim']:.4f})")
        print(f"    Val PSNR:   {val_metrics['psnr']:.2f} dB  (persistence {val_metrics['persistence_psnr']:.2f} dB)")
        print(f"    Time:       {fmt_duration(epoch_secs)} "
              f"(train {fmt_duration(train_secs)}, val {fmt_duration(val_secs)})")
        print(f"    Elapsed:    {fmt_duration(elapsed)}, "
              f"about {fmt_duration(remaining)} left if it runs all {epochs}")

        # Log to wandb
        if config['logging']['use_wandb']:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "lr": current_lr,
                "epoch_seconds": epoch_secs,
                "train_seconds": train_secs,
                "val_seconds": val_secs,
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_ckpt = trainer.save_checkpoint(epoch, val_metrics['loss'])

        # early stopping
        if early.step(val_metrics['loss']):
            print(f"Early stopping at epoch: {epoch}")
            break
        

    total = time.time() - start
    print(f"Training Finished! {epochs_run} epochs in {fmt_duration(total)} "
          f"(average {fmt_duration(total / max(epochs_run, 1))} per epoch)")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # 8. Optionally score the best checkpoint on the held-out test split.
    # Run once, at the end. It stops being an unbiased estimate the moment it
    # is used to make a decision.
    if config['data'].get('evaluate_test'):
        if best_ckpt is None:
            print("No checkpoint improved on the resumed loss -- skipping the test pass.")
        else:
            test_dataset = Clouds(**common, window_range=idx['test'])
            test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)
            print(f"\nScoring {best_ckpt} on the test split "
                  f"({len(test_dataset)} samples, {len(idx['test'])} windows x {len(grid)} crops)")
            trainer.load_checkpoint(best_ckpt)
            test_metrics = trainer.validate(test_loader)
            print(f"    Test Loss: {test_metrics['loss']:.4f}  (persistence {test_metrics['persistence_loss']:.4f})")
            print(f"    Test SSIM: {test_metrics['ssim']:.4f}  (persistence {test_metrics['persistence_ssim']:.4f})")
            print(f"    Test PSNR: {test_metrics['psnr']:.2f} dB  (persistence {test_metrics['persistence_psnr']:.2f} dB)")
            if config['logging']['use_wandb']:
                wandb.log({f"test_{k}": v for k, v in test_metrics.items()})

def parse_args():
    parser = argparse.ArgumentParser(description="Train a cloud-motion forecaster.")
    parser.add_argument("--config", default="config.yaml",
                        help="Config to use. Give each concurrent experiment its own.")
    parser.add_argument("--name", default=None,
                        help="Name for this run (wandb run and checkpoint directory). "
                             "Defaults to <timestamp>_<model>_<arch>.")
    parser.add_argument("--dry-run", type=int, nargs="?", const=5, default=0,
                        metavar="N",
                        help="Run N train and N val steps (default 5), report shapes, "
                             "memory and epoch estimate, then exit without saving or logging.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
