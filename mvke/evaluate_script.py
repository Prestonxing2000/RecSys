import os
import argparse
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle

from utils.data_loader import get_data_loaders, get_feature_configs
from models.mvke_model import MVKEModel, MMoEModel, SharedBottomModel

def get_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    # Load data to get feature configs
    _, _, _, feature_info = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=1,
        num_workers=0
    )
    feature_configs = get_feature_configs(feature_info)
    
    # Create model
    if args.model == 'mvke':
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
    elif args.model == 'mmoe':
        model = MMoEModel(
            feature_configs=feature_configs,
            embedding_dim=args.embedding_dim,
            hidden_dims=[args.hidden_dim, args.hidden_dim//2],
            output_dim=args.output_dim,
            num_experts=args.num_experts,
            dropout=args.dropout
        )
    elif args.model == 'shared_bottom':
        model = SharedBottomModel(
            feature_configs=feature_configs,
            embedding_dim=args.embedding_dim,
            hidden_dims=[args.hidden_dim, args.hidden_dim//2],
            output_dim=args.output_dim,
            dropout=args.dropout
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, args

def evaluate_model(model, data_loader, device):
    """Evaluate model and collect predictions"""
    all_predictions = {'ctr': [], 'cvr': []}
    all_labels = {'ctr': [], 'cvr': []}
    all_vke_weights = {'ctr': [], 'cvr': []} if hasattr(model, 'user_mvke') else None
    
    with torch.no_grad():
        for user_features, item_features, labels in tqdm(data_loader, desc='Evaluating'):
            # Move to device
            user_features = {k: v.to(device) for k, v in user_features.items()}
            item_features = {k: v.to(device) for k, v in item_features.items()}
            labels = {k: v.to(device) for k, v in labels.items()}
            
            # Forward pass
            predictions, auxiliary = model(user_features, item_features, labels)
            
            # Collect predictions
            for task in predictions:
                all_predictions[task].append(torch.sigmoid(predictions[task]).cpu())
                all_labels[task].append(labels[task].cpu())
            
            # Collect VKE weights if available
            if all_vke_weights is not None and 'vke_weights' in auxiliary:
                for task in auxiliary['vke_weights']:
                    all_vke_weights[task].append(auxiliary['vke_weights'][task].cpu())
    
    # Concatenate all batches
    for task in all_predictions:
        all_predictions[task] = torch.cat(all_predictions[task]).numpy()
        all_labels[task] = torch.cat(all_labels[task]).numpy()
    
    if all_vke_weights is not None:
        for task in all_vke_weights:
            if all_vke_weights[task]:
                all_vke_weights[task] = torch.cat(all_vke_weights[task]).numpy()
    
    return all_predictions, all_labels, all_vke_weights

def compute_detailed_metrics(predictions, labels):
    """Compute detailed metrics for each task"""
    metrics = {}
    
    for task in predictions:
        pred = predictions[task].flatten()
        label = labels[task].flatten()
        
        # AUC
        auc = roc_auc_score(label, pred)
        metrics[f'{task}_auc'] = auc
        
        # Average Precision
        ap = average_precision_score(label, pred)
        metrics[f'{task}_ap'] = ap
        
        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(label, pred)
        
        # Find best threshold based on F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        # Compute metrics at best threshold
        pred_binary = (pred >= best_threshold).astype(int)
        tp = np.sum((pred_binary == 1) & (label == 1))
        fp = np.sum((pred_binary == 1) & (label == 0))
        tn = np.sum((pred_binary == 0) & (label == 0))
        fn = np.sum((pred_binary == 0) & (label == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision_best = tp / (tp + fp + 1e-8)
        recall_best = tp / (tp + fn + 1e-8)
        f1_best = 2 * (precision_best * recall_best) / (precision_best + recall_best + 1e-8)
        
        metrics[f'{task}_accuracy'] = accuracy
        metrics[f'{task}_precision'] = precision_best
        metrics[f'{task}_recall'] = recall_best
        metrics[f'{task}_f1'] = f1_best
        metrics[f'{task}_best_threshold'] = best_threshold
        
    return metrics

def plot_results(predictions, labels, vke_weights, output_dir):
    """Generate evaluation plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot ROC curves
    plt.figure(figsize=(10, 5))
    
    for i, task in enumerate(['ctr', 'cvr']):
        plt.subplot(1, 2, i+1)
        
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels[task], predictions[task])
        auc = roc_auc_score(labels[task], predictions[task])
        
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{task.upper()} ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()
    
    # Plot prediction distributions
    plt.figure(figsize=(10, 5))
    
    for i, task in enumerate(['ctr', 'cvr']):
        plt.subplot(1, 2, i+1)
        
        pos_preds = predictions[task][labels[task] == 1]
        neg_preds = predictions[task][labels[task] == 0]
        
        plt.hist(neg_preds, bins=50, alpha=0.5, label='Negative', density=True)
        plt.hist(pos_preds, bins=50, alpha=0.5, label='Positive', density=True)
        plt.xlabel('Prediction Score')
        plt.ylabel('Density')
        plt.title(f'{task.upper()} Prediction Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_distributions.png'))
    plt.close()
    
    # Plot VKE weights if available
    if vke_weights is not None and any(vke_weights[task].size > 0 for task in vke_weights):
        plt.figure(figsize=(10, 5))
        
        for i, task in enumerate(['ctr', 'cvr']):
            if vke_weights[task].size > 0:
                plt.subplot(1, 2, i+1)
                
                # Average weights across samples
                avg_weights = vke_weights[task].mean(axis=0)
                std_weights = vke_weights[task].std(axis=0)
                
                x = np.arange(len(avg_weights))
                plt.bar(x, avg_weights, yerr=std_weights, capsize=5)
                plt.xlabel('Virtual Kernel Index')
                plt.ylabel('Average Weight')
                plt.title(f'{task.upper()} Virtual Kernel Weights')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'vke_weights.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_split', type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, train_args = get_model_from_checkpoint(args.checkpoint_path, device)
    print(f"Loaded {train_args.model} model from {args.checkpoint_path}")
    
    # Load data
    train_loader, val_loader, test_loader, _ = get_data_loaders(
        data_dir=train_args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Select appropriate data loader
    if args.data_split == 'train':
        data_loader = train_loader
    elif args.data_split == 'val':
        data_loader = val_loader
    else:
        data_loader = test_loader
    
    # Evaluate model
    predictions, labels, vke_weights = evaluate_model(model, data_loader, device)
    
    # Compute metrics
    metrics = compute_detailed_metrics(predictions, labels)
    
    # Print results
    print(f"\n{args.data_split.upper()} Set Results:")
    print("-" * 50)
    for metric_name, value in sorted(metrics.items()):
        print(f"{metric_name}: {value:.4f}")
    
    # Save results
    output_dir = os.path.join(args.output_dir, train_args.model)
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'metrics': metrics,
        'predictions': predictions,
        'labels': labels,
        'vke_weights': vke_weights,
        'model_name': train_args.model,
        'checkpoint_path': args.checkpoint_path
    }
    
    with open(os.path.join(output_dir, f'{args.data_split}_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Generate plots
    plot_results(predictions, labels, vke_weights, output_dir)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == '__main__':
    main()