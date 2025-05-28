import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):
    """Embedding layer with optional feature weights"""
    
    def __init__(self, vocab_size, embedding_dim, padding_idx=None):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=padding_idx
        )
        nn.init.normal_(self.embedding.weight, std=0.01)
        
    def forward(self, input_ids, weights=None):
        """
        Args:
            input_ids: [batch_size, seq_len] or [batch_size]
            weights: Optional weights for embeddings
        Returns:
            embeddings: [batch_size, seq_len, embedding_dim] or [batch_size, embedding_dim]
        """
        embeddings = self.embedding(input_ids)
        
        if weights is not None:
            # Apply weights if provided
            if len(weights.shape) == len(input_ids.shape):
                weights = weights.unsqueeze(-1)
            embeddings = embeddings * weights
            
        return embeddings

class MultiValueEmbedding(nn.Module):
    """Embedding layer for multi-valued categorical features"""
    
    def __init__(self, vocab_size, embedding_dim, combiner='mean'):
        super(MultiValueEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.combiner = combiner
        nn.init.normal_(self.embedding.weight, std=0.01)
        
    def forward(self, input_ids, lengths=None):
        """
        Args:
            input_ids: [batch_size, max_seq_len]
            lengths: [batch_size] actual lengths of sequences
        Returns:
            combined_embedding: [batch_size, embedding_dim]
        """
        embeddings = self.embedding(input_ids)  # [batch_size, max_seq_len, embedding_dim]
        
        if lengths is not None:
            # Create mask based on lengths
            mask = torch.arange(input_ids.size(1), device=input_ids.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).float()
            embeddings = embeddings * mask
            
            if self.combiner == 'mean':
                lengths_expanded = lengths.unsqueeze(-1).float()
                combined = embeddings.sum(dim=1) / lengths_expanded.clamp(min=1)
            elif self.combiner == 'sum':
                combined = embeddings.sum(dim=1)
            else:
                raise ValueError(f"Unknown combiner: {self.combiner}")
        else:
            if self.combiner == 'mean':
                combined = embeddings.mean(dim=1)
            elif self.combiner == 'sum':
                combined = embeddings.sum(dim=1)
            else:
                raise ValueError(f"Unknown combiner: {self.combiner}")
                
        return combined

class FeatureEmbedding(nn.Module):
    """Embedding module for all features"""
    
    def __init__(self, feature_configs):
        super(FeatureEmbedding, self).__init__()
        self.embeddings = nn.ModuleDict()
        self.feature_configs = feature_configs
        
        for feat_name, config in feature_configs.items():
            if config['type'] == 'categorical':
                if config.get('multi_value', False):
                    self.embeddings[feat_name] = MultiValueEmbedding(
                        config['vocab_size'],
                        config['embedding_dim'],
                        config.get('combiner', 'mean')
                    )
                else:
                    self.embeddings[feat_name] = EmbeddingLayer(
                        config['vocab_size'],
                        config['embedding_dim']
                    )
            # Numerical features are passed through directly
            
    def forward(self, features):
        """
        Args:
            features: dict of feature_name -> tensor
        Returns:
            embedded_features: dict of feature_name -> embedded tensor
        """
        embedded = {}
        
        for feat_name, feat_value in features.items():
            if feat_name in self.embeddings:
                embedded[feat_name] = self.embeddings[feat_name](feat_value)
            else:
                # Numerical features - just ensure they have embedding dimension
                if self.feature_configs[feat_name]['type'] == 'numerical':
                    if len(feat_value.shape) == 1:
                        feat_value = feat_value.unsqueeze(-1)
                    embedded[feat_name] = feat_value
                    
        return embedded