import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import os

class MultiTaskDataset(Dataset):
    """Dataset for multi-task recommendation"""
    
    def __init__(self, data_path, feature_info_path):
        """
        Args:
            data_path: Path to the pickle file containing the data
            feature_info_path: Path to the feature information pickle file
        """
        # Load data
        self.data = pd.read_pickle(data_path)
        
        # Load feature information
        with open(feature_info_path, 'rb') as f:
            self.feature_info = pickle.load(f)
        
        self.user_features = self.feature_info['user_features']
        self.item_features = self.feature_info['item_features']
        
        # Convert to numpy arrays for faster indexing
        self.user_data = self.data[self.user_features].values
        self.item_data = self.data[self.item_features].values
        self.ctr_labels = self.data['ctr_label'].values
        self.cvr_labels = self.data['cvr_label'].values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        # User features
        user_feats = {}
        for i, feat_name in enumerate(self.user_features):
            user_feats[feat_name] = torch.tensor(self.user_data[idx, i], dtype=torch.long 
                                                if feat_name in ['user_id', 'gender_encoded', 'occupation_encoded'] 
                                                else torch.float32)
        
        # Item features  
        item_feats = {}
        for i, feat_name in enumerate(self.item_features):
            if feat_name == 'item_id':
                item_feats[feat_name] = torch.tensor(self.item_data[idx, i], dtype=torch.long)
            else:
                item_feats[feat_name] = torch.tensor(self.item_data[idx, i], dtype=torch.float32)
        
        # Labels
        labels = {
            'ctr': torch.tensor(self.ctr_labels[idx], dtype=torch.float32),
            'cvr': torch.tensor(self.cvr_labels[idx], dtype=torch.float32)
        }
        
        return user_feats, item_feats, labels

def collate_fn(batch):
    """Custom collate function for batching"""
    user_features_batch = {}
    item_features_batch = {}
    labels_batch = {}
    
    # Initialize
    for user_feats, item_feats, labels in batch:
        for feat_name in user_feats:
            if feat_name not in user_features_batch:
                user_features_batch[feat_name] = []
        for feat_name in item_feats:
            if feat_name not in item_features_batch:
                item_features_batch[feat_name] = []
        for label_name in labels:
            if label_name not in labels_batch:
                labels_batch[label_name] = []
        break
    
    # Collect
    for user_feats, item_feats, labels in batch:
        for feat_name, value in user_feats.items():
            user_features_batch[feat_name].append(value)
        for feat_name, value in item_feats.items():
            item_features_batch[feat_name].append(value)
        for label_name, value in labels.items():
            labels_batch[label_name].append(value)
    
    # Stack
    for feat_name in user_features_batch:
        user_features_batch[feat_name] = torch.stack(user_features_batch[feat_name])
    for feat_name in item_features_batch:
        item_features_batch[feat_name] = torch.stack(item_features_batch[feat_name])
    for label_name in labels_batch:
        labels_batch[label_name] = torch.stack(labels_batch[label_name])
    
    return user_features_batch, item_features_batch, labels_batch

def get_data_loaders(data_dir='data/processed', batch_size=256, num_workers=4):
    """Get data loaders for train, validation, and test sets"""
    
    feature_info_path = os.path.join(data_dir, 'feature_info.pkl')
    
    # Create datasets
    train_dataset = MultiTaskDataset(
        os.path.join(data_dir, 'train.pkl'),
        feature_info_path
    )
    val_dataset = MultiTaskDataset(
        os.path.join(data_dir, 'val.pkl'),
        feature_info_path
    )
    test_dataset = MultiTaskDataset(
        os.path.join(data_dir, 'test.pkl'),
        feature_info_path
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.feature_info

def get_feature_configs(feature_info):
    """Get feature configurations for the model"""
    
    user_configs = {
        'user_id': {
            'type': 'categorical',
            'vocab_size': feature_info['num_users'],
            'embedding_dim': 32
        },
        'age': {
            'type': 'numerical'
        },
        'gender_encoded': {
            'type': 'categorical',
            'vocab_size': 2,
            'embedding_dim': 8
        },
        'occupation_encoded': {
            'type': 'categorical',
            'vocab_size': feature_info['num_occupations'],
            'embedding_dim': 16
        },
        'user_rating_count': {
            'type': 'numerical'
        },
        'user_rating_mean': {
            'type': 'numerical'
        },
        'user_item_count': {
            'type': 'numerical'
        }
    }
    
    item_configs = {
        'item_id': {
            'type': 'categorical',
            'vocab_size': feature_info['num_items'],
            'embedding_dim': 32
        },
        'item_rating_count': {
            'type': 'numerical'
        },
        'item_rating_mean': {
            'type': 'numerical'
        },
        'item_user_count': {
            'type': 'numerical'
        }
    }
    
    # Add genre features
    for i in range(19):
        item_configs[f'genre_{i}'] = {
            'type': 'numerical'
        }
    
    feature_configs = {
        'user': user_configs,
        'item': item_configs
    }
    
    return feature_configs