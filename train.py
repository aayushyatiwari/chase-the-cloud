import torch
import torch.nn as nn
import yaml
import wandb
from torch.utils.data import DataLoader, random_split, Subset
from src.dataset import Clouds
from src.models.convlstm import ConvLSTM
from src.engine import Trainer, EarlyStopping

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1. Load Configuration
    config = load_config('config.yaml')
    
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
    full_dataset = Clouds(
        manifest_path=config['data']['manifest_path'], 
        T=config['data']['T']
    )
    
    train_size = int(config['data']['train_split'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset = Subset(full_dataset, range(0, train_size))
    val_dataset = Subset(full_dataset, range(train_size, len(full_dataset)))
    
    print(f"Dataset loaded. Train samples: {train_size}, Val samples: {val_size}")

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
    model = ConvLSTM(
        input_dim=1, 
        hidden_dim=config['model']['hidden_dim'], 
        kernel_size=config['model']['kernel_size'], 
        num_layers=config['model']['num_layers']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
    criterion = nn.MSELoss()
    early = EarlyStopping(patience=5)
    best_val_loss = float('inf')

    # 5. Initialize the Trainer (The Engine)
    trainer = Trainer(
        model, 
        optimizer, 
        criterion, 
        device, 
        checkpoint_dir=config['train']['checkpoint_dir']
    )
    
    # 6. The Main Training Loop
    print("Starting Training...")
    epochs = config['train']['epochs']

    for epoch in range(1, epochs + 1):
        # Train
        avg_train_loss = trainer.train_one_epoch(train_loader, epoch)
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        
        print(f"==> Epoch {epoch} Complete.")
        print(f"    Train Loss: {avg_train_loss:.4f}")
        print(f"    Val Loss:   {val_metrics['loss']:.4f}")
        print(f"    Val SSIM:   {val_metrics['ssim']:.4f}")
        print(f"    Val CSI:    {val_metrics['csi']:.4f}")
        
        # Log to wandb
        if config['logging']['use_wandb']:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_metrics['loss'],
                "val_ssim": val_metrics['ssim'],
                "val_csi": val_metrics['csi']
            })

        # Save every N epochs
        # if epoch % config['train']['save_every'] == 0:
        #     trainer.save_checkpoint(epoch, val_metrics['loss'])

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            trainer.save_checkpoint(epoch, val_metrics['loss']) 

        # early stopping
        if early.step(val_metrics['loss']):
            print(f"Early stopping at epoch: {epoch}")
            break
        

    print("Training Finished!")

if __name__ == "__main__":
    main()
