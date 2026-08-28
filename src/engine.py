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
            # 1. Prepare data: (Batch, Time, H, W) -> (Batch, Time, 1, H, W)
            # We add a 'Channel' dimension of 1 because Conv2d expects [B, C, H, W]
            inputs = inputs.unsqueeze(2).to(self.device)
            targets = targets.unsqueeze(1).to(self.device)
            
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

    def validate(self, dataloader, threshold=0.5):
        """
        Runs a pass through the validation data without updating weights.

        Also scores a persistence forecast (repeat the last input frame) on the
        same batches. A model that does not clearly beat persistence has learned
        no cloud motion, so these numbers belong next to every model metric.
        """
        from src.utils import ssim, csi_counts, csi_from_counts
        self.model.eval()
        # 'model' and 'persistence' each track [loss, ssim] sums and pooled CSI counts
        sums = {'model': [0.0, 0.0], 'persistence': [0.0, 0.0]}
        counts = {'model': [0.0, 0.0, 0.0], 'persistence': [0.0, 0.0, 0.0]}

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.unsqueeze(2).to(self.device)
                targets = targets.unsqueeze(1).to(self.device)

                preds = {
                    'model': self.model(inputs),
                    'persistence': inputs[:, -1],
                }

                for name, pred in preds.items():
                    sums[name][0] += self.criterion(pred, targets).item()
                    sums[name][1] += ssim(pred, targets).item()
                    for i, c in enumerate(csi_counts(pred, targets, threshold=threshold)):
                        counts[name][i] += c

        n = len(dataloader)
        return {
            'loss': sums['model'][0] / n,
            'ssim': sums['model'][1] / n,
            'csi': csi_from_counts(*counts['model']),
            'persistence_loss': sums['persistence'][0] / n,
            'persistence_ssim': sums['persistence'][1] / n,
            'persistence_csi': csi_from_counts(*counts['persistence']),
        }

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
    def __init__(self, patience=5, min_delta=0.001):
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