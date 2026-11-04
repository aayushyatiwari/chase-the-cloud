import torch
import os
from datetime import datetime


def compute_loss(model, criterion, inputs, targets):
    """
    The training loss for one batch.

    A regressor is scored on its output: criterion(model(inputs), targets). A
    diffusion model is not -- its forward() runs the whole sampler, and what it
    trains on is a noise-prediction loss -- so it supplies training_loss itself.
    """
    if hasattr(model, 'training_loss'):
        return model.training_loss(inputs, targets)
    return criterion(model(inputs), targets)


def after_step(model):
    """Post-optimizer-step hook; the DDPM uses it to update its EMA weights."""
    if hasattr(model, 'after_step'):
        model.after_step()

class Trainer:
    """
    The Engine: This class handles the actual training loop, validation, and checkpointing.
    Keeping this separate from train.py makes your code much cleaner.
    """
    def __init__(self, model, optimizer, criterion, device, checkpoint_dir='checkpoints', run_name=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        # Each run writes into its own timestamped subdirectory, so checkpoints
        # from different runs (and different architectures) never interleave
        # under the same model_epoch_N.pt names.
        self.run_name = run_name or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.checkpoint_dir = os.path.join(checkpoint_dir, self.run_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"Checkpoints -> {self.checkpoint_dir}")

    def train_one_epoch(self, dataloader, epoch):
        """Runs one full pass through the training data."""
        self.model.train()
        # Weighted by batch size, like validate: the last batch is usually
        # short and should not count as much as a full one.
        running_loss = 0.0
        n_seen = 0
        
        for i, (inputs, targets) in enumerate(dataloader):
            # The dataset already provides the channel axis:
            # inputs (B, T, C, H, W), targets (B, C, H, W).
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # 2. Forward pass
            self.optimizer.zero_grad()
            loss = compute_loss(self.model, self.criterion, inputs, targets)

            # 3. Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            after_step(self.model)
            
            running_loss += loss.item() * targets.shape[0]
            n_seen += targets.shape[0]
            
            # Log progress every 10 batches
            if i % 10 == 0:
                print(f"Epoch [{epoch}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        return running_loss / n_seen

    def validate(self, dataloader):
        """
        Runs a pass through the validation data without updating weights.

        Also scores a persistence forecast (repeat the last input frame) on the
        same batches. A model that does not clearly beat persistence has learned
        no cloud motion, so these numbers belong next to every model metric.

        Runs under a fixed, forked RNG. The regressors are deterministic in eval
        and do not notice; a diffusion model samples, and without this its
        validation score would change from epoch to epoch through the noise
        alone -- and consume the training RNG stream while doing it.
        """
        devices = [self.device] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(0)
            return self._validate(dataloader)

    def _validate(self, dataloader):
        from src.utils import ssim, squared_error_counts, psnr_from_mse
        self.model.eval()

        # Loss and SSIM are means over a batch, so they are accumulated weighted
        # by batch size -- the last batch is usually short and must not count as
        # much as a full one.
        #
        # PSNR is not a mean at all. It is pooled as squared error and elements,
        # then turned into decibels once at the end, for the reason spelled out
        # in psnr_from_mse: averaging per-batch PSNR averages logarithms.
        weighted = {'model': [0.0, 0.0], 'persistence': [0.0, 0.0]}   # [loss, ssim]
        sq_err = {'model': 0.0, 'persistence': 0.0}
        n_elem = {'model': 0, 'persistence': 0}
        n_seen = 0
        # Diffusion only: the training objective on held-out data. Cheap and
        # smooth, but only loosely tied to how good the samples are.
        denoise = 0.0
        is_diffusion = hasattr(self.model, 'training_loss')

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                batch = targets.shape[0]
                n_seen += batch

                # Persistence repeats the last input frame, but only the
                # channels being predicted -- with extra input channels (e.g.
                # water vapour) the inputs are wider than the target, and the
                # predicted channel (TIR1) comes first.
                preds = {
                    'model': self.model(inputs),
                    'persistence': inputs[:, -1, :targets.shape[1]],
                }

                # Unclipped: every metric scores what the model actually
                # produced. The range is handled by normalising the inputs.
                for name, pred in preds.items():
                    weighted[name][0] += self.criterion(pred, targets).item() * batch
                    weighted[name][1] += ssim(pred, targets).item() * batch
                    se, n = squared_error_counts(pred, targets)
                    sq_err[name] += se
                    n_elem[name] += n
                if is_diffusion:
                    denoise += self.model.training_loss(inputs, targets).item() * batch

        metrics = {}
        for name in ('model', 'persistence'):
            prefix = '' if name == 'model' else 'persistence_'
            metrics[prefix + 'loss'] = weighted[name][0] / n_seen
            metrics[prefix + 'ssim'] = weighted[name][1] / n_seen
            metrics[prefix + 'psnr'] = psnr_from_mse(sq_err[name] / n_elem[name])
        if is_diffusion:
            metrics['denoise_loss'] = denoise / n_seen
        return metrics


    def load_checkpoint(self, path, lr=None):
        """
        Restore model and optimizer state so training can continue.

        Restoring the optimizer matters as much as the weights: Adam's first and
        second moment buffers are part of the search state, and starting a fresh
        optimizer over loaded weights causes a visible loss spike while they
        rebuild. Returns (completed_epoch, val_loss) from the checkpoint.

        load_state_dict also restores the learning rate that was saved, so pass
        `lr` to override it -- that is how you resume at a lower rate.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if lr is not None:
            for group in self.optimizer.param_groups:
                group['lr'] = lr

        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))
        print(f"Resumed from {path} (epoch {epoch}, val_loss {loss:.4f}, lr {self.optimizer.param_groups[0]['lr']})")
        return epoch, loss

    def save_checkpoint(self, epoch, loss):
        """Saves model weights to disk."""
        path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)
        print(f"--- Saved checkpoint: {path} ---")
        return path


class EarlyStopping:
    """
    A simple early stopping mechanism to prevent overfitting.
    """
    def __init__(self, patience=10, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # Not stopping
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered. No improvement in {self.patience} epochs.")
                return True  # Stop training
            return False  # Not stopping yet