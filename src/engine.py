import torch
import os

class Trainer:
    """
    The Engine: This class handles the actual training loop, validation, and checkpointing.
    Keeping this separate from train.py makes your code much cleaner.
    """
    def __init__(self, model, optimizer, criterion, device, checkpoint_dir='checkpoints'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        # self.early_stopping = EarlyStopping(patience=5)
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)

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
        """Runs a pass through the validation data without updating weights."""
        from src.utils import ssim, calculate_csi
        self.model.eval()
        val_loss = 0.0
        val_ssim = 0.0
        val_csi = 0.0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.unsqueeze(2).to(self.device)
                targets = targets.unsqueeze(1).to(self.device)
                
                outputs = self.model(inputs)
                
                # Calculate MSE (Loss)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                
                # Calculate SSIM
                val_ssim += ssim(outputs, targets).item()
                
                # Calculate CSI
                val_csi += calculate_csi(outputs, targets, threshold=threshold).item()
                
        metrics = {
            'loss': val_loss / len(dataloader),
            'ssim': val_ssim / len(dataloader),
            'csi': val_csi / len(dataloader)
        }
        return metrics

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