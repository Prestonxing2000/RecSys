import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from utils.data_loader import get_data_loaders, get_feature_configs
from models.mvke_model import MVKEModel, MMoEModel, SharedBottomModel

# 添加安全全局变量
torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])

def get_model(model_name, feature_configs, device, args):
    """Get model instance based on name"""
    if model_name == 'mvke':
        model = MVKEModel(
            feature_configs=feature_configs,
            embedding_dim=args.embedding_dim,
            hidden_dims=[args.hidden_dim, args.hidden_dim//2],
            output_dim=args.output_dim,
            num_vk=args.num_vk,
            shared_experts=args.shared_experts,
            dropout=args.dropout,
            task_config={
                'ctr': (0, 2),
                'cvr': (1, 4)
            }
        )
    elif model_name == 'mmoe':
        model = MMoEModel(
            feature_configs=feature_configs,
            embedding_dim=args.embedding_dim,
            hidden_dims=[args.hidden_dim, args.hidden_dim//2],
            output_dim=args.output_dim,
            num_experts=args.num_experts,
            dropout=args.dropout
        )
    elif model_name == 'shared_bottom':
        model = SharedBottomModel(
            feature_configs=feature_configs,
            embedding_dim=args.embedding_dim,
            hidden_dims=[args.hidden_dim, args.hidden_dim//2],
            output_dim=args.output_dim,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device)

def compute_metrics(predictions, labels):
    """Compute AUC metrics for each task"""
    metrics = {}
    
    for task in predictions:
        pred = torch.sigmoid(predictions[task]).detach().cpu().numpy()
        label = labels[task].detach().cpu().numpy()
        
        # Compute AUC
        try:
            auc = roc_auc_score(label, pred)
            metrics[f'{task}_auc'] = auc
        except:
            metrics[f'{task}_auc'] = 0.5
            
    return metrics

def train_epoch(model, train_loader, optimizer, device, epoch, writer):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    task_losses = {'ctr': 0, 'cvr': 0}
    all_predictions = {'ctr': [], 'cvr': []}
    all_labels = {'ctr': [], 'cvr': []}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (user_features, item_features, labels) in enumerate(pbar):
        # Move to device
        user_features = {k: v.to(device) for k, v in user_features.items()}
        item_features = {k: v.to(device) for k, v in item_features.items()}
        labels = {k: v.to(device) for k, v in labels.items()}
        
        # Forward pass
        predictions, auxiliary = model(user_features, item_features, labels)
        loss = auxiliary['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record losses
        total_loss += loss.item()
        for task in task_losses:
            task_losses[task] += auxiliary['task_losses'][task].item()
        
        # Record predictions
        for task in predictions:
            all_predictions[task].append(predictions[task].detach())
            all_labels[task].append(labels[task].detach())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'ctr_loss': auxiliary['task_losses']['ctr'].item(),
            'cvr_loss': auxiliary['task_losses']['cvr'].item()
        })
        
        # Log to tensorboard
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('train/batch_loss', loss.item(), global_step)
    
    # Compute epoch metrics
    for task in all_predictions:
        all_predictions[task] = torch.cat(all_predictions[task])
        all_labels[task] = torch.cat(all_labels[task])
    
    metrics = compute_metrics(all_predictions, all_labels)
    avg_loss = total_loss / len(train_loader)
    
    # Log epoch metrics
    writer.add_scalar('train/epoch_loss', avg_loss, epoch)
    for task in task_losses:
        writer.add_scalar(f'train/{task}_loss', task_losses[task] / len(train_loader), epoch)
    for metric_name, value in metrics.items():
        writer.add_scalar(f'train/{metric_name}', value, epoch)
    
    return avg_loss, metrics

def validate(model, val_loader, device, epoch, writer):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    task_losses = {'ctr': 0, 'cvr': 0}
    all_predictions = {'ctr': [], 'cvr': []}
    all_labels = {'ctr': [], 'cvr': []}
    
    with torch.no_grad():
        for user_features, item_features, labels in tqdm(val_loader, desc='Validation'):
            # Move to device
            user_features = {k: v.to(device) for k, v in user_features.items()}
            item_features = {k: v.to(device) for k, v in item_features.items()}
            labels = {k: v.to(device) for k, v in labels.items()}
            
            # Forward pass
            predictions, auxiliary = model(user_features, item_features, labels)
            loss = auxiliary['loss']
            
            # Record losses
            total_loss += loss.item()
            for task in task_losses:
                task_losses[task] += auxiliary['task_losses'][task].item()
            
            # Record predictions
            for task in predictions:
                all_predictions[task].append(predictions[task])
                all_labels[task].append(labels[task])
    
    # Compute metrics
    for task in all_predictions:
        all_predictions[task] = torch.cat(all_predictions[task])
        all_labels[task] = torch.cat(all_labels[task])
    
    metrics = compute_metrics(all_predictions, all_labels)
    avg_loss = total_loss / len(val_loader)
    
    # Log metrics
    writer.add_scalar('val/epoch_loss', avg_loss, epoch)
    for task in task_losses:
        writer.add_scalar(f'val/{task}_loss', task_losses[task] / len(val_loader), epoch)
    for metric_name, value in metrics.items():
        writer.add_scalar(f'val/{metric_name}', value, epoch)
    
    return avg_loss, metrics

def train(args):
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader, feature_info = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Get feature configurations
    feature_configs = get_feature_configs(feature_info)
    print(f"Feature configs: {feature_configs}")
    print(f"User feature count: {len(feature_configs['user'])}")
    
    # Create model
    model = get_model(args.model, feature_configs, device, args)
    print(f"Model: {args.model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        min_lr=1e-6
    )
    
    # Create tensorboard writer
    log_dir = os.path.join(args.log_dir, f"{args.model}_{args.experiment_name}")
    writer = SummaryWriter(log_dir)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.checkpoint_dir, f"{args.model}_{args.experiment_name}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_val_auc = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, writer)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, device, epoch, writer)
        
        # Compute average validation AUC
        val_auc = (val_metrics['ctr_auc'] + val_metrics['cvr_auc']) / 2
        
        # Update learning rate
        scheduler.step(val_auc)
        
        # Print results
        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}, CTR AUC: {train_metrics['ctr_auc']:.4f}, CVR AUC: {train_metrics['cvr_auc']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, CTR AUC: {val_metrics['ctr_auc']:.4f}, CVR AUC: {val_metrics['cvr_auc']:.4f}")
        
        # Save checkpoint if best
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'args': args
            }
            
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best model with validation AUC: {val_auc:.4f}")
        
        # Early stopping
        if epoch - best_epoch > args.early_stopping_patience:
            print(f"Early stopping triggered. Best epoch: {best_epoch}")
            break
    
    # Test the best model
    print("\nTesting best model...")
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_metrics = validate(model, test_loader, device, epoch, writer)
    print(f"Test Loss: {test_loss:.4f}, CTR AUC: {test_metrics['ctr_auc']:.4f}, CVR AUC: {test_metrics['cvr_auc']:.4f}")
    
    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'best_epoch': best_epoch,
        'best_val_metrics': checkpoint['val_metrics']
    }
    
    import pickle
    with open(os.path.join(checkpoint_dir, 'test_results.pkl'), 'wb') as f:
        pickle.dump(test_results, f)
    
    writer.close()
    print("Training completed!")

def main():
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument('--model', type=str, default='mvke', 
                        choices=['mvke', 'mmoe', 'shared_bottom'])
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--output_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # MVKE specific arguments
    parser.add_argument('--num_vk', type=int, default=5, help='Number of virtual kernels')
    parser.add_argument('--shared_experts', action='store_true', 
                        help='Whether to use shared experts in MVKE')
    
    # MMoE specific arguments
    parser.add_argument('--num_experts', type=int, default=10, help='Number of experts in MMoE')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--experiment_name', type=str, default='default')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    train(args)

if __name__ == '__main__':
    main()