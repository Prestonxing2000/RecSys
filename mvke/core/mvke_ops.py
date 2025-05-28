import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MVKEBottomLayer(nn.Module):
    """MVKE bottom layer with attention mechanism"""
    
    def __init__(self, feature_dim, vk_num):
        super(MVKEBottomLayer, self).__init__()
        self.feature_dim = feature_dim
        self.vk_num = vk_num
        
        # Projection layers for attention
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
        
        # Initialize weights
        nn.init.normal_(self.query_proj.weight, std=0.01)
        nn.init.normal_(self.key_proj.weight, std=0.01)
        nn.init.normal_(self.value_proj.weight, std=0.01)
        
    def forward(self, vk_emb, shared_field_embs):
        """
        Args:
            vk_emb: [vk_num, emb_size] - virtual kernel embeddings
            shared_field_embs: [batch_size, feature_num, emb_size] - feature embeddings
        Returns:
            output: [batch_size, feature_num, vk_num, emb_size]
        """
        batch_size, feature_num, emb_size = shared_field_embs.shape
        
        # Project virtual kernels as queries
        query = self.query_proj(vk_emb)  # [vk_num, emb_size]
        
        # Project features as keys and values
        reshaped_features = shared_field_embs.reshape(-1, emb_size)
        key = self.key_proj(reshaped_features)  # [batch_size * feature_num, emb_size]
        value = self.value_proj(reshaped_features)  # [batch_size * feature_num, emb_size]
        
        # Compute attention scores
        query_t = query.t()  # [emb_size, vk_num]
        scores = torch.matmul(key, query_t) / math.sqrt(float(emb_size))
        scores = scores.reshape(batch_size, feature_num, self.vk_num)  # [batch_size, feature_num, vk_num]
        
        # Apply softmax over features dimension
        attention_weights = F.softmax(scores, dim=1)  # [batch_size, feature_num, vk_num]
        attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, feature_num, vk_num, 1]
        
        # Apply attention to values
        value = value.reshape(batch_size, feature_num, 1, emb_size)
        value_expanded = value.expand(-1, -1, self.vk_num, -1)  # [batch_size, feature_num, vk_num, emb_size]
        
        # Weighted values
        weighted_values = attention_weights * value_expanded
        
        # Add residual connection and layer norm
        residual = shared_field_embs.unsqueeze(2).expand(-1, -1, self.vk_num, -1)
        output = self.layer_norm(weighted_values + residual)
        
        return output

class VKGLayer(nn.Module):
    """Virtual Kernel Gating layer"""
    
    def __init__(self, emb_size):
        super(VKGLayer, self).__init__()
        self.emb_size = emb_size
        
    def forward(self, tag_emb, vk_emb, vke_outputs):
        """
        Args:
            tag_emb: [batch_size, emb_size] - item embeddings
            vk_emb: [vk_num, emb_size] - virtual kernel embeddings
            vke_outputs: [batch_size, vk_num, emb_size] - outputs from virtual kernel experts
        Returns:
            output: [batch_size, emb_size]
            weights: [batch_size, vk_num]
        """
        vk_num = vk_emb.shape[0]
        
        # Compute attention scores between items and virtual kernels
        vk_emb_t = vk_emb.t()  # [emb_size, vk_num]
        scores = torch.matmul(tag_emb, vk_emb_t)  # [batch_size, vk_num]
        weights = F.softmax(scores, dim=1)  # [batch_size, vk_num]
        
        # Apply weights to expert outputs
        weights_expanded = weights.unsqueeze(1)  # [batch_size, 1, vk_num]
        output = torch.matmul(weights_expanded, vke_outputs)  # [batch_size, 1, emb_size]
        output = output.squeeze(1)  # [batch_size, emb_size]
        
        return output, weights

class MVKEModule(nn.Module):
    """Complete MVKE module"""
    
    def __init__(self, feature_dim, hidden_dims, output_dim, vk_num=5, 
                 shared_experts=False, dropout=0.1, num_features=None):
        super(MVKEModule, self).__init__()
        self.feature_dim = feature_dim
        self.vk_num = vk_num
        self.shared_experts = shared_experts
        self.num_features = num_features
        
        # Virtual kernels
        self.virtual_kernels = nn.Parameter(
            torch.randn(vk_num, feature_dim) * 0.01
        )
        
        # MVKE bottom layer
        self.mvke_bottom = MVKEBottomLayer(feature_dim, vk_num)
        
        # Calculate the actual input dimension for experts
        # Each expert takes flattened features: num_features * feature_dim
        if num_features is None:
            raise ValueError("num_features must be specified for MVKEModule")
        
        expert_input_dim = num_features * feature_dim
        
        # Expert networks
        if shared_experts:
            # Shared expert architecture
            self.expert = self._build_expert_network(
                expert_input_dim, hidden_dims, output_dim, dropout
            )
        else:
            # Separate experts for each virtual kernel
            self.experts = nn.ModuleList([
                self._build_expert_network(expert_input_dim, hidden_dims, output_dim, dropout)
                for _ in range(vk_num)
            ])
        
        # VKG layer
        self.vkg_layer = VKGLayer(output_dim)
        
    def _build_expert_network(self, input_dim, hidden_dims, output_dim, dropout):
        """Build a single expert network"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, feature_embeddings, tag_embeddings, task_config):
        """
        Args:
            feature_embeddings: [batch_size, feature_num, emb_size]
            tag_embeddings: dict of task_name -> [batch_size, emb_size]
            task_config: dict of task_name -> (start_idx, end_idx) for VK selection
        Returns:
            outputs: dict of task_name -> [batch_size, output_dim]
            vke_weights: dict of task_name -> [batch_size, selected_vk_num]
        """
        batch_size, feature_num, emb_size = feature_embeddings.shape
        
        # MVKE bottom layer
        mvke_features = self.mvke_bottom(self.virtual_kernels, feature_embeddings)
        # mvke_features: [batch_size, feature_num, vk_num, emb_size]
        
        # Process through experts
        if self.shared_experts:
            # Reshape for shared expert
            mvke_reshaped = mvke_features.permute(0, 2, 1, 3).contiguous()
            mvke_reshaped = mvke_reshaped.reshape(batch_size * self.vk_num, feature_num * emb_size)
            expert_outputs = self.expert(mvke_reshaped)
            expert_outputs = expert_outputs.reshape(batch_size, self.vk_num, -1)
        else:
            # Process each VK through its own expert
            expert_outputs = []
            for vk_idx in range(self.vk_num):
                vk_features = mvke_features[:, :, vk_idx, :]  # [batch_size, feature_num, emb_size]
                vk_features = vk_features.reshape(batch_size, feature_num * emb_size)
                expert_out = self.experts[vk_idx](vk_features)
                expert_outputs.append(expert_out)
            expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, vk_num, output_dim]
        
        # Apply VKG for each task
        outputs = {}
        vke_weights = {}
        
        for task_name, (start_idx, end_idx) in task_config.items():
            # Select relevant virtual kernels
            selected_vk = self.virtual_kernels[start_idx:end_idx+1]
            selected_experts = expert_outputs[:, start_idx:end_idx+1, :]
            
            # Apply VKG layer
            task_output, task_weights = self.vkg_layer(
                tag_embeddings[task_name], 
                selected_vk, 
                selected_experts
            )
            
            outputs[task_name] = task_output
            vke_weights[task_name] = task_weights
            
        return outputs, vke_weights

class MMoEModule(nn.Module):
    """Multi-gate Mixture of Experts module"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, num_experts=10, 
                 num_tasks=2, dropout=0.1):
        super(MMoEModule, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        
        # Expert networks
        self.experts = nn.ModuleList([
            self._build_expert_network(input_dim, hidden_dims, output_dim, dropout)
            for _ in range(num_experts)
        ])
        
        # Gate networks for each task
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=1)
            )
            for _ in range(num_tasks)
        ])
        
    def _build_expert_network(self, input_dim, hidden_dims, output_dim, dropout):
        """Build a single expert network"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, inputs, task_names):
        """
        Args:
            inputs: [batch_size, input_dim]
            task_names: list of task names
        Returns:
            outputs: dict of task_name -> [batch_size, output_dim]
        """
        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(inputs))
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, output_dim]
        
        # Apply gates for each task
        outputs = {}
        for i, task_name in enumerate(task_names):
            # Compute gate weights
            gate_weights = self.gates[i](inputs)  # [batch_size, num_experts]
            gate_weights = gate_weights.unsqueeze(1)  # [batch_size, 1, num_experts]
            
            # Weighted sum of expert outputs
            task_output = torch.matmul(gate_weights, expert_outputs)  # [batch_size, 1, output_dim]
            outputs[task_name] = task_output.squeeze(1)  # [batch_size, output_dim]
            
        return outputs