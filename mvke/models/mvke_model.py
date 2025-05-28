import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.embedding_ops import FeatureEmbedding
from core.mvke_ops import MVKEModule, MMoEModule

class BaseMultiTaskModel(nn.Module):
    """Base class for multi-task recommendation models"""
    
    def __init__(self, feature_configs, embedding_dim=64):
        super(BaseMultiTaskModel, self).__init__()
        self.feature_configs = feature_configs
        self.embedding_dim = embedding_dim
        
        # Feature embeddings
        self.user_embedding = FeatureEmbedding(feature_configs['user'])
        self.item_embedding = FeatureEmbedding(feature_configs['item'])
        
    def compute_loss(self, predictions, labels, loss_weights=None):
        """Compute multi-task loss"""
        if loss_weights is None:
            loss_weights = {task: 1.0 for task in predictions.keys()}
            
        total_loss = 0
        task_losses = {}
        
        for task, pred in predictions.items():
            task_loss = F.binary_cross_entropy_with_logits(
                pred.squeeze(), 
                labels[task].float()
            )
            task_losses[task] = task_loss
            total_loss += loss_weights[task] * task_loss
            
        return total_loss, task_losses
    
    def forward(self, user_features, item_features, labels=None):
        raise NotImplementedError

class MVKEModel(BaseMultiTaskModel):
    """MVKE model for multi-objective user profile modeling"""
    
    def __init__(self, feature_configs, embedding_dim=64, hidden_dims=[256, 128], 
                 output_dim=64, num_vk=5, shared_experts=False, dropout=0.1,
                 task_config=None):
        super(MVKEModel, self).__init__(feature_configs, embedding_dim)
        
        # Default task configuration
        if task_config is None:
            task_config = {
                'ctr': (0, 2),  # Use VK 0-2 for CTR
                'cvr': (1, 4)   # Use VK 1-4 for CVR
            }
        self.task_config = task_config
        self.tasks = list(task_config.keys())
        
        # Calculate input dimensions
        self.user_input_dim = self._calculate_input_dim(feature_configs['user'])
        self.item_input_dim = self._calculate_input_dim(feature_configs['item'])
        
        # Count number of user features
        num_user_features = len(feature_configs['user'])
        
        # User tower with MVKE
        self.user_mvke = MVKEModule(
            feature_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            vk_num=num_vk,
            shared_experts=shared_experts,
            dropout=dropout,
            num_features=num_user_features
        )
        
        # Item tower with MMoE (hard-sharing as in paper)
        self.item_mmoe = MMoEModule(
            input_dim=self.item_input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_experts=len(self.tasks),  # Separate expert for each task
            num_tasks=len(self.tasks),
            dropout=dropout
        )
        
        # Task-specific projection layers
        self.task_projections = nn.ModuleDict({
            task: nn.Linear(embedding_dim, embedding_dim)
            for task in self.tasks
        })
        
    def _calculate_input_dim(self, feature_config):
        """Calculate total input dimension for a feature group"""
        total_dim = 0
        for feat_name, config in feature_config.items():
            if config['type'] == 'categorical':
                total_dim += config['embedding_dim']
            else:  # numerical
                total_dim += 1
        return total_dim
    
    def _aggregate_embeddings(self, embedded_features):
        """Aggregate multiple feature embeddings"""
        # Concatenate all embeddings
        embeddings = []
        for feat_name, embedding in embedded_features.items():
            if len(embedding.shape) == 2:
                embeddings.append(embedding)
            else:
                embeddings.append(embedding.squeeze(1))
        
        return torch.cat(embeddings, dim=-1)
    
    def forward(self, user_features, item_features, labels=None):
        """
        Args:
            user_features: dict of feature_name -> tensor
            item_features: dict of feature_name -> tensor
            labels: dict of task_name -> tensor
        Returns:
            predictions: dict of task_name -> logits
            auxiliary_outputs: dict with additional outputs for analysis
        """
        batch_size = next(iter(user_features.values())).shape[0]
        
        # Embed features
        user_embedded = self.user_embedding(user_features)
        item_embedded = self.item_embedding(item_features)
        
        # Process user features for MVKE
        # Create feature matrix [batch_size, num_features, embedding_dim]
        user_feature_list = []
        for feat_name, embedding in user_embedded.items():
            if len(embedding.shape) == 2:
                embedding = embedding.unsqueeze(1)  # [batch_size, 1, dim]
            # Pad to embedding_dim if necessary
            if embedding.shape[-1] < self.embedding_dim:
                padding = torch.zeros(
                    batch_size, 1, self.embedding_dim - embedding.shape[-1],
                    device=embedding.device
                )
                embedding = torch.cat([embedding, padding], dim=-1)
            elif embedding.shape[-1] > self.embedding_dim:
                embedding = embedding[:, :, :self.embedding_dim]
            user_feature_list.append(embedding)
        
        user_feature_matrix = torch.cat(user_feature_list, dim=1)
        
        # Process item features for MMoE
        item_aggregated = self._aggregate_embeddings(item_embedded)
        
        # Get item embeddings for each task
        item_task_embeddings = self.item_mmoe(item_aggregated, self.tasks)
        
        # Process item embeddings through task projections
        item_task_embeddings_proj = {
            task: self.task_projections[task](emb)
            for task, emb in item_task_embeddings.items()
        }
        
        # Get user embeddings using MVKE
        user_task_embeddings, vke_weights = self.user_mvke(
            user_feature_matrix,
            item_task_embeddings_proj,
            self.task_config
        )
        
        # Compute predictions for each task
        predictions = {}
        for task in self.tasks:
            user_emb = user_task_embeddings[task]
            item_emb = item_task_embeddings[task]
            
            # Dot product for prediction
            logits = torch.sum(user_emb * item_emb, dim=1, keepdim=True)
            predictions[task] = logits
        
        # Compute loss if labels provided
        if labels is not None:
            total_loss, task_losses = self.compute_loss(predictions, labels)
            return predictions, {
                'loss': total_loss,
                'task_losses': task_losses,
                'vke_weights': vke_weights
            }
        
        return predictions, {'vke_weights': vke_weights}

class MMoEModel(BaseMultiTaskModel):
    """MMoE baseline model"""
    
    def __init__(self, feature_configs, embedding_dim=64, hidden_dims=[256, 128],
                 output_dim=64, num_experts=10, dropout=0.1):
        super(MMoEModel, self).__init__(feature_configs, embedding_dim)
        
        self.tasks = ['ctr', 'cvr']
        
        # Calculate input dimensions
        self.user_input_dim = self._calculate_input_dim(feature_configs['user'])
        self.item_input_dim = self._calculate_input_dim(feature_configs['item'])
        
        # User tower with MMoE
        self.user_mmoe = MMoEModule(
            input_dim=self.user_input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_experts=num_experts,
            num_tasks=len(self.tasks),
            dropout=dropout
        )
        
        # Item tower with MMoE
        self.item_mmoe = MMoEModule(
            input_dim=self.item_input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_experts=2,  # Separate expert for each task
            num_tasks=len(self.tasks),
            dropout=dropout
        )
        
    def _calculate_input_dim(self, feature_config):
        """Calculate total input dimension for a feature group"""
        total_dim = 0
        for feat_name, config in feature_config.items():
            if config['type'] == 'categorical':
                total_dim += config['embedding_dim']
            else:  # numerical
                total_dim += 1
        return total_dim
    
    def _aggregate_embeddings(self, embedded_features):
        """Aggregate multiple feature embeddings"""
        embeddings = []
        for feat_name, embedding in embedded_features.items():
            if len(embedding.shape) == 2:
                embeddings.append(embedding)
            else:
                embeddings.append(embedding.squeeze(1))
        
        return torch.cat(embeddings, dim=-1)
    
    def forward(self, user_features, item_features, labels=None):
        """Forward pass"""
        # Embed features
        user_embedded = self.user_embedding(user_features)
        item_embedded = self.item_embedding(item_features)
        
        # Aggregate embeddings
        user_aggregated = self._aggregate_embeddings(user_embedded)
        item_aggregated = self._aggregate_embeddings(item_embedded)
        
        # Get task-specific embeddings
        user_task_embeddings = self.user_mmoe(user_aggregated, self.tasks)
        item_task_embeddings = self.item_mmoe(item_aggregated, self.tasks)
        
        # Compute predictions
        predictions = {}
        for task in self.tasks:
            user_emb = user_task_embeddings[task]
            item_emb = item_task_embeddings[task]
            
            # Dot product for prediction
            logits = torch.sum(user_emb * item_emb, dim=1, keepdim=True)
            predictions[task] = logits
        
        # Compute loss if labels provided
        if labels is not None:
            total_loss, task_losses = self.compute_loss(predictions, labels)
            return predictions, {
                'loss': total_loss,
                'task_losses': task_losses
            }
        
        return predictions, {}

class SharedBottomModel(BaseMultiTaskModel):
    """Shared bottom baseline model"""
    
    def __init__(self, feature_configs, embedding_dim=64, hidden_dims=[256, 128],
                 output_dim=64, dropout=0.1):
        super(SharedBottomModel, self).__init__(feature_configs, embedding_dim)
        
        self.tasks = ['ctr', 'cvr']
        
        # Calculate input dimensions
        self.user_input_dim = self._calculate_input_dim(feature_configs['user'])
        self.item_input_dim = self._calculate_input_dim(feature_configs['item'])
        
        # Shared bottom layers
        self.user_shared = self._build_shared_layers(
            self.user_input_dim, hidden_dims, dropout
        )
        self.item_shared = self._build_shared_layers(
            self.item_input_dim, hidden_dims, dropout
        )
        
        # Task-specific heads
        self.user_heads = nn.ModuleDict({
            task: nn.Linear(hidden_dims[-1], output_dim)
            for task in self.tasks
        })
        self.item_heads = nn.ModuleDict({
            task: nn.Linear(hidden_dims[-1], output_dim)
            for task in self.tasks
        })
        
    def _calculate_input_dim(self, feature_config):
        """Calculate total input dimension for a feature group"""
        total_dim = 0
        for feat_name, config in feature_config.items():
            if config['type'] == 'categorical':
                total_dim += config['embedding_dim']
            else:  # numerical
                total_dim += 1
        return total_dim
    
    def _build_shared_layers(self, input_dim, hidden_dims, dropout):
        """Build shared bottom layers"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
            
        return nn.Sequential(*layers)
    
    def _aggregate_embeddings(self, embedded_features):
        """Aggregate multiple feature embeddings"""
        embeddings = []
        for feat_name, embedding in embedded_features.items():
            if len(embedding.shape) == 2:
                embeddings.append(embedding)
            else:
                embeddings.append(embedding.squeeze(1))
        
        return torch.cat(embeddings, dim=-1)
    
    def forward(self, user_features, item_features, labels=None):
        """Forward pass"""
        # Embed features
        user_embedded = self.user_embedding(user_features)
        item_embedded = self.item_embedding(item_features)
        
        # Aggregate embeddings
        user_aggregated = self._aggregate_embeddings(user_embedded)
        item_aggregated = self._aggregate_embeddings(item_embedded)
        
        # Shared bottom
        user_shared = self.user_shared(user_aggregated)
        item_shared = self.item_shared(item_aggregated)
        
        # Task-specific heads
        predictions = {}
        for task in self.tasks:
            user_emb = self.user_heads[task](user_shared)
            item_emb = self.item_heads[task](item_shared)
            
            # Dot product for prediction
            logits = torch.sum(user_emb * item_emb, dim=1, keepdim=True)
            predictions[task] = logits
        
        # Compute loss if labels provided
        if labels is not None:
            total_loss, task_losses = self.compute_loss(predictions, labels)
            return predictions, {
                'loss': total_loss,
                'task_losses': task_losses
            }
        
        return predictions, {}