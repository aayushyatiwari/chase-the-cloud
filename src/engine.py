import torch
import os
from datetime import datetime

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
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(dataloader):
            # The dataset already provides the channel axis:
            # inputs (B, T, C, H, W), targets (B, C, H, W).
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # 2. Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 3. Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Log progress every 10 batches
            if i % 10 == 0:
                print(f"Epoch [{epoch}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        return running_loss / len(dataloader)

    def validate(self, dataloader):
        """
        Runs a pass through the validation data without updating weights.

        Also scores a persistence forecast (repeat the last input frame) on the
        same batches. A model that does not clearly beat persistence has learned
        no cloud motion, so these numbers belong next to every model metric.
        """
        from src.utils import ssim, psnr
        self.model.eval()
        # each entry is [loss, ssim, psnr] summed over batches
        sums = {'model': [0.0, 0.0, 0.0], 'persistence': [0.0, 0.0, 0.0]}

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Persistence repeats the last input frame, but only the
                # channels being predicted -- with extra input channels (e.g.
                # water vapour) the inputs are wider than the target, and the
                # predicted channel (TIR1) comes first.
                preds = {
                    'model': self.model(inputs),
                    'persistence': inputs[:, -1, :targets.shape[1]],
                }

                for name, pred in preds.items():
                    # Loss is on the raw output, since that is what training
                    # actually minimises.
                    sums[name][0] += self.criterion(pred, targets).item()
                    # SSIM and PSNR assume values in [0,1], but the output layer
                    # is unbounded and drifts slightly outside it, so clip first.
                    # The SimVP reference implementation clips the same way.
                    clipped = pred.clamp(0.0, 1.0)
                    sums[name][1] += ssim(clipped, targets).item()
                    sums[name][2] += psnr(clipped, targets).item()

        n = len(dataloader)
        return {
            'loss': sums['model'][0] / n,
            'ssim': sums['model'][1] / n,
            'psnr': sums['model'][2] / n,
            'persistence_loss': sums['persistence'][0] / n,
            'persistence_ssim': sums['persistence'][1] / n,
            'persistence_psnr': sums['persistence'][2] / n,
        }

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