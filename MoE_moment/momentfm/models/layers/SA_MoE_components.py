import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertDefine(nn.Module):
    
    def __init__(self,d_model,d_ff,dropout=0.1,expert_id=0,num_experts=4,expert_dim_ratio=1.0):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.expert_id = expert_id
        self.num_experts = num_experts

        self.expert_dim = max(64, int(d_ff * expert_dim_ratio))
        self.expert_dim_ratio = expert_dim_ratio
        
        self.wi_0 = nn.Linear(d_model, self.expert_dim, bias=False)
        self.wi_1 = nn.Linear(d_model, self.expert_dim, bias=False)
        self.wo = nn.Linear(self.expert_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        
    def forward(self, freq_feature):
        original_shape = freq_feature.shape

        # [batch, seq_len, d_model]
        if len(original_shape) == 2:
            freq_feature = freq_feature.unsqueeze(0)

        enhanced_freq = freq_feature

        if len(original_shape) == 2:
            enhanced_freq = enhanced_freq.squeeze(0)

        hidden_gelu = self.act(self.wi_0(enhanced_freq))
        hidden_linear = self.wi_1(enhanced_freq)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        output = self.wo(hidden_states)

        return output


class SharedExpertPool(nn.Module):
    def __init__(self,d_model,d_ff,num_experts=4,dropout=0.1,
                 expert_dim_ratio=0.5,max_freq=40.0,sampling_rate=256.0):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.expert_dim_ratio = expert_dim_ratio
        # self.max_freq = max_freq
        # self.sampling_rate = sampling_rate
        
        self.global_experts = nn.ModuleList([
            ExpertDefine(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                expert_id=i,
                num_experts=num_experts,
                expert_dim_ratio=expert_dim_ratio
            ) for i in range(num_experts)
        ])
        
        self.shared_router = None
        
        self.registered_layers = set()
        
        self.expert_usage_stats = {
            'total_calls': torch.zeros(num_experts),
            'layer_calls': {},
        }
        
    def register_layer(self, layer_id):
        if layer_id not in self.registered_layers:
            self.registered_layers.add(layer_id)
            self.expert_usage_stats['layer_calls'][layer_id] = torch.zeros(self.num_experts)
            
    def get_expert(self, expert_id):
        return self.global_experts[expert_id]
    
    def forward(self, expert_id, freq_features, layer_id=None):
        # [batch, seq_len, d_model, num_experts]
        if self.training:
            self.expert_usage_stats['total_calls'][expert_id] += 1
            if layer_id is not None and layer_id in self.expert_usage_stats['layer_calls']:
                self.expert_usage_stats['layer_calls'][layer_id][expert_id] += 1

        # [batch, seq_len, d_model]
        expert_freq_feature = freq_features[:, :, :, expert_id]

        expert = self.global_experts[expert_id]
        expert_output = expert(expert_freq_feature)

        # [num_tokens, d_model]
        if expert_output.dim() == 3 and expert_output.shape[1] == 1:
            expert_output = expert_output.squeeze(1)

        return expert_output
    
    def get_parameter_efficiency_report(self):
        expert_params = sum(p.numel() for expert in self.global_experts for p in expert.parameters())
        
        router_params = sum(p.numel() for p in self.shared_router.parameters())
        
        total_global_params = expert_params + router_params
        
        num_layers = len(self.registered_layers)
        independent_expert_params = expert_params * num_layers
        independent_router_params = router_params * num_layers
        total_independent_params = independent_expert_params + independent_router_params
        
        parameter_savings = total_independent_params - total_global_params
        reduction_ratio = parameter_savings / total_independent_params if total_independent_params > 0 else 0
        
        return {
            'global_expert_params': expert_params,
            'shared_router_params': router_params,
            'total_global_pool_params': total_global_params,
            'independent_expert_params': independent_expert_params,
            'independent_router_params': independent_router_params,
            'total_independent_params': total_independent_params,
            'parameter_savings': parameter_savings,
            'parameter_reduction_ratio': reduction_ratio,
            'num_registered_layers': num_layers,
            'experts_per_layer': self.num_experts,
            'memory_efficiency_score': reduction_ratio * 100
        }
    
    def get_usage_statistics(self):
        stats = {
            'total_expert_calls': self.expert_usage_stats['total_calls'].clone(),
            'expert_utilization_ratio': self.expert_usage_stats['total_calls'] / (self.expert_usage_stats['total_calls'].sum() + 1e-8),
            'registered_layers': sorted(list(self.registered_layers)),
            'layer_wise_usage': {}
        }
        
        for layer_id, layer_calls in self.expert_usage_stats['layer_calls'].items():
            stats['layer_wise_usage'][layer_id] = {
                'calls': layer_calls.clone(),
                'utilization': layer_calls / (layer_calls.sum() + 1e-8)
            }
        
        return stats
    
    def reset_statistics(self):
        self.expert_usage_stats['total_calls'].zero_()
        for layer_calls in self.expert_usage_stats['layer_calls'].values():
            layer_calls.zero_()
    
    def freeze_experts(self):
        for expert in self.global_experts:
            for param in expert.parameters():
                param.requires_grad = False
                
    def unfreeze_experts(self):
        for expert in self.global_experts:
            for param in expert.parameters():
                param.requires_grad = True
    
    def freeze_router(self):
        if self.shared_router is not None:
            for param in self.shared_router.parameters():
                param.requires_grad = False

    def unfreeze_router(self):
        if self.shared_router is not None:
            for param in self.shared_router.parameters():
                param.requires_grad = True
    
    def analyze_frequency_specialization(self, x):
        analysis = {}
        
        for expert_id, expert in enumerate(self.global_experts):
            if hasattr(expert, 'freq_processor') and expert.freq_processor is not None:
                if hasattr(expert.freq_processor, 'freq_weights'):
                    freq_weights = expert.freq_processor.freq_weights
                    analysis[f'pure_global_expert_{expert_id}'] = {
                        'freq_weights': freq_weights.detach(),
                        'dominant_freq_idx': torch.argmax(freq_weights).item(),
                        'freq_specialization_score': (torch.max(freq_weights) / (torch.mean(freq_weights) + 1e-8)).item(),
                        'freq_distribution': torch.softmax(freq_weights, dim=0).detach()
                    }
        
        return analysis


class Router(nn.Module):
    def __init__(self,d_model,num_experts,top_k=2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.frequency_router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_experts)
        )

    def forward(self, hidden_states, layer_id=0):
        logits = self.frequency_router(hidden_states)

        router_probs = F.softmax(logits, dim=-1)

        top_k_gates, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        top_k_gates = top_k_gates / (top_k_gates.sum(dim=-1, keepdim=True) + 1e-8)

        return top_k_gates, top_k_indices, router_probs


