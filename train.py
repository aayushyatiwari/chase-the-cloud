import json
import random
import shutil
import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import yaml
import wandb
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from src.dataset import Clouds, tile_grid
from src.models.convlstm import ConvLSTM
from src.models.simvp import SimVP
from src.models.residual import BoundedOutput, ResidualWrapper
from src.engine import Trainer, EarlyStopping
from src.utils import latest_checkpoint
import time

def use_shm_safe_sharing(min_gb=1.0):
    """
    Fall back to file-based tensor sharing when /dev/shm is small.

    DataLoader workers hand tensors to the main process through shared memory.
    Containers routinely ship a 64MB /dev/shm, which overflows and kills the
    workers with a bus error -- and it surfaces as a RuntimeError inside the
    model's forward pass, which points at entirely the wrong code. The
    file_system strategy passes tensors through temp files instead, so this
    needs no change to how the container was started.

    Only applied when /dev/shm is actually small: the default strategy is
    faster, and on a normal machine there is nothing to work around.
    """
    try:
        free_gb = shutil.disk_usage('/dev/shm').total / 1e9
    except OSError:
        return
    if free_gb < min_gb:
        torch.multiprocessing.set_sharing_strategy('file_system')
        print(f"/dev/shm is only {free_gb:.2f}GB -- "
              "using file_system tensor sharing so DataLoader workers survive")


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

def main():
    # 1. Load Configuration
    config = load_config('config.yaml')
    set_seed(config['train']['seed'])
    use_shm_safe_sharing()

    # 2. Initialize wandb
    if config['logging']['use_wandb']:
        wandb.init(
            project=config['logging']['project'],
            config=config # log hyperparameters
        )
    
    # 3. Hardware Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 3. Data Preparation
    # Split by time window, then let each side pick its own crops. Splitting
    # windows first keeps every crop of a timestamp on one side of the boundary.
    # The T-window gap means no raw frame is shared across the split, because
    # the manifest slides one frame at a time.
    T = config['data']['T']
    crop_size = config['data']['crop_size']
    n_windows = len(json.load(open(config['data']['manifest_path'])))
    train_size = int(config['data']['train_split'] * n_windows)

    common = dict(manifest_path=config['data']['manifest_path'], T=T, crop_size=crop_size)

    # Training crops move around every epoch, which is the whole point of
    # multi-crop: same number of steps, but new regions each time.
    train_dataset = Clouds(**common, window_range=range(0, train_size), random_crop=True)

    # Validation uses a fixed grid so the metric measures the model, not which
    # crops happened to come up. val_stride thins the grid to keep it quick.
    grid = tile_grid(train_dataset.H, train_dataset.W, crop_size,
                     stride=config['data']['val_crop_stride'])
    val_dataset = Clouds(**common, window_range=range(train_size + T, n_windows), crops=grid)

    print(f"Frames are {train_dataset.C}x{train_dataset.H}x{train_dataset.W}, crop {crop_size}")
    print(f"Train: {len(train_dataset)} samples ({train_size} windows, 1 random crop each)")
    print(f"Val:   {len(val_dataset)} samples ({n_windows - train_size - T} windows x {len(grid)} crops)")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=True, 
        num_workers=config['train']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=False, 
        num_workers=config['train']['num_workers']
    )
    
    # 4. Initialize Model, Optimizer, and Loss Function
    # Channel count comes from the data, so adding channels needs no code change.
    C_in = train_dataset.C
    model_type = config['model']['type']
    if model_type == 'convlstm':
        model = ConvLSTM(
            input_dim=C_in,
            hidden_dim=config['model']['hidden_dim'],
            kernel_size=config['model']['kernel_size'],
            num_layers=config['model']['num_layers']
        ).to(device)
        arch_tag = f"L{config['model']['num_layers']}_h{config['model']['hidden_dim']}"
    elif model_type == 'simvp':
        model = SimVP(
            shape_in=(T, C_in, crop_size, crop_size),
            hid_S=config['model']['hid_S'],
            hid_T=config['model']['hid_T'],
            N_S=config['model']['N_S'],
            N_T=config['model']['N_T'],
            T_out=1,
            groups=config['model']['groups'],
        ).to(device)
        arch_tag = f"hidS{config['model']['hid_S']}_NT{config['model']['N_T']}"
    else:
        raise ValueError(f"Unknown model.type: {model_type!r} (expected 'convlstm' or 'simvp')")

    # Predict the change from the last frame rather than the frame itself.
    # Either way the final prediction is bounded to [0,1], so the two modes are
    # trained against the same bounded objective and can be compared. They are
    # bounded differently because they have to be -- see BoundedOutput.
    if config['model'].get('residual'):
        model = ResidualWrapper(model, out_channels=1).to(device)
        arch_tag += "_res"
        print("Residual mode: model predicts the change from the last input frame")
    else:
        model = BoundedOutput(model).to(device)


    # float() guards against YAML parsing e.g. 3e-5 as a string
    lr = float(config['train']['lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

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

    early = EarlyStopping(
        patience=config['train'].get('early_stopping_patience', 10),
        # float() guards against YAML reading 1.0e-5 as a string
        min_delta=float(config['train'].get('early_stopping_min_delta', 1e-5)),
    )
    best_val_loss = float('inf')

    # 5. Initialize the Trainer (The Engine)
    # e.g. 20260828_161422_convlstm_L3_h64 -- timestamp plus architecture, so a
    # stale checkpoint can never be mistaken for one matching the current config.
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_type}_{arch_tag}"

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
            resume_from = latest_checkpoint(config['train']['checkpoint_dir'])
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
            trainer.save_checkpoint(epoch, val_metrics['loss']) 

        # early stopping
        if early.step(val_metrics['loss']):
            print(f"Early stopping at epoch: {epoch}")
            break
        

    total = time.time() - start
    print(f"Training Finished! {epochs_run} epochs in {fmt_duration(total)} "
          f"(average {fmt_duration(total / max(epochs_run, 1))} per epoch)")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
