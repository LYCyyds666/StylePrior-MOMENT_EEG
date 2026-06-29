import torch
import torch.nn as nn
import torch.nn.functional as F
from .SA_MoE_components import SharedExpertPool, Router


class StyleAdaptor(nn.Module):
    def __init__(self,num_channels,feature_dim,initial_gamma=None,initial_beta=None):
        super().__init__()

        if initial_gamma is not None:
            gamma_val = initial_gamma
        else:
            gamma_val = torch.ones(num_channels, feature_dim)

        if initial_beta is not None:
            beta_val = initial_beta
        else:
            beta_val = torch.zeros(num_channels, feature_dim)

        self.gamma = nn.Parameter(gamma_val.clone())
        self.beta = nn.Parameter(beta_val.clone())

    def forward(self):
        return self.gamma, self.beta


class ExpertEmbeddings(nn.Module):
    def __init__(self,num_experts,expert_embedding_dim):
        super().__init__()
        self.num_experts = num_experts
        self.expert_embedding_dim = expert_embedding_dim

        self.expert_embeddings = nn.Embedding(num_experts, expert_embedding_dim)

        nn.init.normal_(self.expert_embeddings.weight, mean=0.0, std=0.02)

    def forward(self,expert_indices):
        return self.expert_embeddings(expert_indices)

    def get_selection_embedding(self,router_probs,top_k_indices):

        mask = torch.ones_like(router_probs)

        mask.scatter_(-1, top_k_indices, 0.0)

        unselected_probs = router_probs * mask
        unselected_probs_sum = torch.sum(unselected_probs, dim=-1, keepdim=True)

        unselected_probs_normalized = unselected_probs / (unselected_probs_sum + 1e-8)

        selection_embedding = torch.matmul(unselected_probs_normalized, self.expert_embeddings.weight)

        return selection_embedding

    def get_remaining_experts_embedding(self,activated_expert_indices):

        device = activated_expert_indices.device
        return torch.zeros(self.expert_embedding_dim, device=device)


class SharedBackboneHyperNetwork(nn.Module):
    def __init__(self,subject_embedding_dim,expert_embedding_dim,feature_dim,num_channels,hidden_dim=128,moe_conditioning_dim=64):
        super().__init__()
        self.subject_embedding_dim = subject_embedding_dim
        self.expert_embedding_dim = expert_embedding_dim
        self.feature_dim = feature_dim
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.moe_conditioning_dim = moe_conditioning_dim

        self.shared_backbone = nn.Sequential(
            nn.Linear(subject_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        style_output_dim = num_channels * feature_dim * 2
        self.style_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, style_output_dim)
        )

        moe_input_dim = hidden_dim + expert_embedding_dim
        self.moe_head = nn.Sequential(
            nn.Linear(moe_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, moe_conditioning_dim)
        )

    def forward(self,subject_embeddings,selection_embedding):
        batch_size = subject_embeddings.shape[0]

        global_style_context = self.shared_backbone(subject_embeddings)

        style_params = self.style_head(global_style_context)

        style_params = style_params.view(batch_size, self.num_channels, self.feature_dim, 2)
        gamma = style_params[:, :, :, 0]
        beta = style_params[:, :, :, 1]

        gamma = F.softplus(gamma) + 1e-8

        moe_input = torch.cat([global_style_context, selection_embedding], dim=-1)
        moe_conditioning_vector = self.moe_head(moe_input)

        return {
            'gamma': gamma,
            'beta': beta,
            'moe_conditioning_vector': moe_conditioning_vector,
            'global_style_context': global_style_context
        }


class SharedHyperExpert(nn.Module):
    def __init__(self,feature_dim,conditioning_dim,hidden_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.conditioning_dim = conditioning_dim
        self.hidden_dim = hidden_dim

        input_dim = feature_dim + conditioning_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.01)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self,x,c):
        combined_input = torch.cat([x, c], dim=-1)
        return self.net(combined_input)


class SA_MoE(nn.Module):
    def __init__(self,d_model,d_ff,layer_id,optimized_global_pool,shared_router,num_experts=4,
                 top_k=2,dropout=0.1,aux_loss_weight=0.01, 
                 enable_shared_backbone_hypernetwork=False,num_subjects=50,
                 subject_embedding_dim=64,expert_embedding_dim=32,hyper_expert_hidden_dim=64,
                 num_channels=16,moe_conditioning_dim=64):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.layer_id = layer_id
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.enable_shared_backbone_hypernetwork = enable_shared_backbone_hypernetwork
        self.num_channels = num_channels
        self.num_subjects = num_subjects
        self.subject_embedding_dim = subject_embedding_dim
        self.expert_embedding_dim = expert_embedding_dim
        self.hyper_expert_hidden_dim = hyper_expert_hidden_dim
        self.moe_conditioning_dim = moe_conditioning_dim

        self.optimized_global_pool = optimized_global_pool

        self.shared_router = shared_router

        self.layer_norm = nn.LayerNorm(d_model)

        if enable_shared_backbone_hypernetwork:
            self.expert_embeddings = ExpertEmbeddings(
                num_experts=num_experts,
                expert_embedding_dim=expert_embedding_dim
            )

            self.subject_embedding = nn.Embedding(num_subjects, subject_embedding_dim)
            nn.init.normal_(self.subject_embedding.weight, mean=0.0, std=0.02)

            self.shared_backbone_hypernetwork = SharedBackboneHyperNetwork(
                subject_embedding_dim=subject_embedding_dim,
                expert_embedding_dim=expert_embedding_dim,
                feature_dim=d_model,
                num_channels=num_channels,
                hidden_dim=128,
                moe_conditioning_dim=moe_conditioning_dim
            )

            self.shared_hyper_expert = SharedHyperExpert(
                feature_dim=d_model,
                conditioning_dim=moe_conditioning_dim,
                hidden_dim=hyper_expert_hidden_dim
            )
        else:
            self.expert_embeddings = None
            self.subject_embedding = None
            self.shared_backbone_hypernetwork = None
            self.shared_hyper_expert = None
        
        self.aux_losses = []
        
        self.is_frozen = False
        
        self.optimized_global_pool.register_layer(layer_id)
        
        self._last_gates = None

        self._current_subject_ids = None

        self.in_tta_mode = False
        self.tta_adaptor = None
        self.spatio_temporal_generator = None

        self.in_STSA_tta_mode = False
        self._last_raw_features = None
        self._last_norm_features = None
        self._last_gamma = None
        self._last_beta = None

    def freeze_parameters(self):
        self.is_frozen = True
        for param in self.layer_norm.parameters():
            param.requires_grad = False
            
    def unfreeze_parameters(self):
        self.is_frozen = False
        for param in self.layer_norm.parameters():
            param.requires_grad = True

    def freeze_shared_components(self):
        if self.optimized_global_pool is not None:
            self.optimized_global_pool.freeze_experts()
            self.optimized_global_pool.freeze_router()

    def unfreeze_shared_components(self):
        if self.optimized_global_pool is not None:
            self.optimized_global_pool.unfreeze_experts()
            self.optimized_global_pool.unfreeze_router()

    def set_subject_ids(self,subject_ids):
        self._current_subject_ids = subject_ids

    def switch_to_tta_mode(self,adaptor):
        self.in_tta_mode = True
        self.tta_adaptor = adaptor

    def switch_to_pretrain_mode(self):
        self.in_tta_mode = False
        self.tta_adaptor = None
        self.spatio_temporal_generator = None
        self.in_STSA_tta_mode = False
        self._last_raw_features = None
        self._last_norm_features = None
        self._last_gamma = None
        self._last_beta = None

    def switch_to_STSA(self,adaptor=None):
        self.in_STSA_tta_mode = True
        self.in_tta_mode = True
        if adaptor is not None:
            self.tta_adaptor = adaptor

        self.spatio_temporal_generator = None

    def get_STSA_tta_features(self):
        if (self.in_STSA_tta_mode and
            self.in_tta_mode and
            self.tta_adaptor is not None and
            self._last_raw_features is not None and
            self._last_norm_features is not None and
            self._last_gamma is not None and
            self._last_beta is not None):
            return (self._last_raw_features, self._last_norm_features,
                    self._last_gamma, self._last_beta)
        return None, None, None, None

    def forward(self,hidden_states,subject_ids=None):
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)

        channels_times_batch, seq_len, d_model = hidden_states.shape

        actual_batch_size = len(subject_ids)
        num_channels = channels_times_batch // actual_batch_size

        current_subject_ids = subject_ids if subject_ids is not None else self._current_subject_ids

        if self.enable_shared_backbone_hypernetwork and current_subject_ids is not None:
            result = self._forward_with_shared_backbone_hypernetwork(
                hidden_states, current_subject_ids, residual,
                channels_times_batch, seq_len, d_model, actual_batch_size
            )

            if isinstance(result, dict):
                self._last_raw_features = result["raw_features"]
                self._last_norm_features = result["norm_features"]
                self._last_gamma = result["gamma_used"]
                self._last_beta = result["beta_used"]
                return result["output"]
            else:
                return result

    def _forward_with_shared_backbone_hypernetwork(self,hidden_states,subject_ids,
                                                   residual,channels_times_batch,seq_len,
                                                   d_model,actual_batch_size):
        zero_based_subject_ids = torch.clamp(subject_ids - 1, 0, self.subject_embedding.num_embeddings - 1)
        subject_embeddings = self.subject_embedding(zero_based_subject_ids)

        if actual_batch_size is None:
            actual_batch_size = subject_embeddings.shape[0]

        B, C, S, F = actual_batch_size, self.num_channels, seq_len, d_model

        global_style_context = self.shared_backbone_hypernetwork.shared_backbone(subject_embeddings)

        if self.in_tta_mode and self.tta_adaptor is not None:
            gamma_single, beta_single = self.tta_adaptor()

            w = 0.8
            with torch.no_grad():
                style_params = self.shared_backbone_hypernetwork.style_head(global_style_context)
                style_params_reshaped = style_params.view(B, C, 2 * F)
                original_gamma, original_beta = torch.chunk(style_params_reshaped, 2, dim=-1)
                original_gamma = torch.nn.functional.softplus(original_gamma) + 1e-8
                original_beta = original_beta.detach()
                original_gamma = original_gamma.detach()

            # alpha=0.2: give more weight to new subject style (tuned for cross-subject)
            gamma = (1.0 - w) * original_gamma + w * gamma_single.unsqueeze(0)
            beta  = (1.0 - w) * original_beta  + w * beta_single.unsqueeze(0)

            if self.in_STSA_tta_mode and self.in_tta_mode:
                hidden_states_reshaped = hidden_states.view(channels_times_batch, seq_len, d_model)
                normalized_features = torch.nn.functional.instance_norm(
                    hidden_states_reshaped.permute(0, 2, 1)
                ).permute(0, 2, 1)
        else:
            style_params = self.shared_backbone_hypernetwork.style_head(global_style_context)
            style_params_reshaped = style_params.view(B, C, 2 * F)
            gamma, beta = torch.chunk(style_params_reshaped, 2, dim=-1)
            gamma = torch.nn.functional.softplus(gamma) + 1e-8

        aligned_features = self._apply_style_alignment(
            hidden_states, gamma, beta, actual_batch_size, channels_times_batch, seq_len, d_model
        )
        
        # aligned_features = hidden_states  # BUG FIX: style alignment now applied

        router_output = self.shared_router(aligned_features, layer_id=self.layer_id)

        if len(router_output) == 3:
            gates, indices, router_probs = router_output
            self._last_router_probs = router_probs
        else:
            gates, indices = router_output
            self._last_router_probs = None
            router_probs = torch.nn.functional.softmax(gates.sum(dim=-1, keepdim=True).expand(-1, -1, self.num_experts), dim=-1)

        self._last_gates = gates
        self._last_indices = indices

        router_probs_flat = router_probs.view(-1, self.num_experts)
        indices_flat = indices.view(-1, self.top_k)

        selection_embedding_flat = self.expert_embeddings.get_selection_embedding(
            router_probs_flat, indices_flat
        )

        global_style_context_flat = global_style_context.unsqueeze(1).expand(-1, C * S, -1).contiguous().view(-1, global_style_context.shape[-1])

        moe_head_input = torch.cat([global_style_context_flat, selection_embedding_flat], dim=-1)
        conditioning_vectors_flat = self.shared_backbone_hypernetwork.moe_head(moe_head_input)

        aligned_flat = aligned_features.view(-1, F)
        gates_flat = gates.view(-1, self.top_k)

        final_expert_output_flat = self._calculate_moe_output(
            aligned_flat, gates_flat, indices_flat, conditioning_vectors_flat
        )

        output = final_expert_output_flat.view(channels_times_batch, seq_len, d_model)
        output = residual + output

        if self.training and self.aux_loss_weight > 0:
            aux_loss = self._calculate_aux_loss(gates)
            self.aux_losses.append(aux_loss)

        if self.in_STSA_tta_mode and self.in_tta_mode:
            return {
                "output": output,
                "raw_features": hidden_states,
                "norm_features": normalized_features,
                "gamma_used": gamma,
                "beta_used": beta,
                "layer_id": self.layer_id
            }
        else:
            return output

    def _apply_style_alignment(self,hidden_states,gamma,beta,actual_batch_size,channels_times_batch,seq_len,d_model):
        hidden_states_4d = hidden_states.view(self.num_channels, actual_batch_size, seq_len, d_model)
        hidden_states_4d = hidden_states_4d.permute(1, 0, 2, 3)

        B, C, S, feat_dim = hidden_states_4d.shape
        h_to_norm = hidden_states_4d.contiguous().view(B * C, S, feat_dim)
        h_to_norm_permuted = h_to_norm.permute(0, 2, 1)

        h_norm = F.instance_norm(h_to_norm_permuted, eps=1e-8)
        h_norm = h_norm.permute(0, 2, 1).view(B, C, S, feat_dim)

        gamma_expanded = gamma.unsqueeze(2)
        beta_expanded = beta.unsqueeze(2)

        aligned_features_4d = h_norm * gamma_expanded + beta_expanded

        aligned_features_4d = aligned_features_4d.permute(1, 0, 2, 3)
        aligned_features = aligned_features_4d.contiguous().view(channels_times_batch, seq_len, d_model)

        return aligned_features

    def _calculate_moe_output(self,aligned_flat,gates_flat,indices_flat,conditioning_vectors):
        """BUG FIX: HSE now applied per paper Eq.(4)"""
        final_output = torch.zeros_like(aligned_flat)

        for k in range(self.top_k):
            expert_indices = indices_flat[:, k]
            gate_values = gates_flat[:, k].unsqueeze(-1)

            for expert_idx in range(self.num_experts):
                expert_mask = (expert_indices == expert_idx)
                if expert_mask.any():
                    token_indices = torch.where(expert_mask)[0]
                    if len(token_indices) > 0:
                        selected_features = aligned_flat[token_indices]
                        selected_cond     = conditioning_vectors[token_indices]

                        expert = self.optimized_global_pool.get_expert(expert_idx)
                        main_expert_output = expert(selected_features)

                        # BUG FIX: add HSE contribution (paper Eq.4)
                        hse_output = self.shared_hyper_expert(selected_features, selected_cond)
                        combined_output = main_expert_output + hse_output

                        weighted_output = gate_values[token_indices] * combined_output
                        final_output[token_indices] += weighted_output

        return final_output

    def _calculate_aux_loss(self,gates):
        if hasattr(self, '_last_router_probs') and self._last_router_probs is not None:
            return self._calculate_switch_aux_loss()
        raise NotImplementedError("not implement")

    def _calculate_switch_aux_loss(self):
        router_probs = self._last_router_probs
        indices = self._last_indices

        num_experts = router_probs.shape[-1]

        router_probs_flat = router_probs.view(-1, num_experts)
        indices_flat = indices.view(-1, self.top_k)

        P_i = torch.mean(router_probs_flat, dim=0)

        expert_mask = F.one_hot(indices_flat, num_classes=num_experts).float()

        tokens_per_expert = torch.mean(expert_mask, dim=0)

        overall_loss = torch.sum(tokens_per_expert * P_i.unsqueeze(0))

        aux_loss = self.aux_loss_weight * num_experts * overall_loss

        return aux_loss
    
    def get_aux_loss(self):
        if self.aux_losses:
            total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
            for loss in self.aux_losses:
                total_loss = total_loss + loss
            return total_loss / len(self.aux_losses)
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def clear_aux_losses(self):
        self.aux_losses.clear()
    
    def get_expert_stats(self):
        base_stats = {}
        
        global_stats = self.optimized_global_pool.get_usage_statistics()
        
        if self.layer_id in global_stats['layer_wise_usage']:
            layer_usage = global_stats['layer_wise_usage'][self.layer_id]
            base_stats.update({
                'layer_expert_calls': layer_usage['calls'],
                'layer_expert_utilization': layer_usage['utilization']
            })
        
        base_stats.update({
            'module_type': 'SA_MoE',
            'layer_id': self.layer_id,
            'is_frozen': self.is_frozen,
            'num_frequency_experts': self.num_experts,
            'top_k': self.top_k,
            'local_parameters': sum(p.numel() for p in self.layer_norm.parameters()),
            'trainable_local_parameters': sum(p.numel() for p in self.layer_norm.parameters() if p.requires_grad),
            'optimized_global_pool_stats': global_stats
        })
        
        return base_stats
    
    def analyze_frequency_specialization(self,x):
        return self.optimized_global_pool.analyze_frequency_specialization(x)
    
    def get_knowledge_representation(self,x):
        with torch.no_grad():
            output = self.forward(x)
            
            expert_stats = self.get_expert_stats()
            freq_analysis = self.analyze_frequency_specialization(x)
            
            return {
                'shared_features': output,
                'expert_activations': expert_stats,
                'frequency_analysis': freq_analysis,
                'invariance_score': self._compute_invariance_score(output),
                'optimized_pool_efficiency': self.optimized_global_pool.get_parameter_efficiency_report()
            }
    
    def _compute_invariance_score(self,features):
        feature_std = torch.std(features, dim=(0, 1))
        invariance_score = 1.0 / (1.0 + feature_std.mean())
        
        return invariance_score
    
    def get_parameter_efficiency_metrics(self):
        global_efficiency = self.optimized_global_pool.get_parameter_efficiency_report()
        
        local_params = sum(p.numel() for p in self.layer_norm.parameters())
        
        return {
            'local_module_params': local_params,
            'optimized_pool_efficiency': global_efficiency,
            'total_shared_experts': global_efficiency['global_expert_params'],
            'shared_router_params': global_efficiency['shared_router_params'],
            'parameter_reduction_ratio': global_efficiency['parameter_reduction_ratio'],
            'parameter_savings': global_efficiency['parameter_savings'],
            'memory_efficiency_score': global_efficiency['memory_efficiency_score'],
            'layer_specific_overhead': local_params
        }


class SA_MoEFactory:
    def __init__(self,d_model,d_ff,num_experts=4,dropout=0.1,freq_learning_mode="adaptive_filter",
                 routing_strategy="frequency_aware",expert_dim_ratio=1.0,max_freq=40.0,sampling_rate=256.0):
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.dropout = dropout
        self.freq_learning_mode = freq_learning_mode
        self.routing_strategy = routing_strategy
        self.expert_dim_ratio = expert_dim_ratio
        self.max_freq = max_freq
        self.sampling_rate = sampling_rate
        
        self.optimized_global_pool = SharedExpertPool(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            dropout=dropout,
            expert_dim_ratio=expert_dim_ratio,
            max_freq=self.max_freq,
            sampling_rate=sampling_rate
        )

        self.shared_router = None
        
        self.created_modules = []
        
    def create_module(self,layer_id,top_k=2,aux_loss_weight=0.01,enable_subject_style_normalization=None,
                      enable_shared_backbone_hypernetwork=False,num_subjects=50,subject_embedding_dim=64,
                      expert_embedding_dim=32,hyper_expert_hidden_dim=64,num_channels=16,moe_conditioning_dim=64):
        if self.shared_router is None:
            self.shared_router = Router(
                d_model=self.d_model,
                num_experts=self.num_experts,
                top_k=top_k
            )
                
            self.optimized_global_pool.shared_router = self.shared_router

        if hasattr(self.shared_router, 'top_k') and self.shared_router.top_k != top_k:
            top_k = self.shared_router.top_k

        module = SA_MoE(
            d_model=self.d_model,
            d_ff=self.d_ff,
            layer_id=layer_id,
            optimized_global_pool=self.optimized_global_pool,
            shared_router=self.shared_router,
            num_experts=self.num_experts,
            top_k=top_k,
            dropout=self.dropout,
            aux_loss_weight=aux_loss_weight,
            enable_shared_backbone_hypernetwork=enable_shared_backbone_hypernetwork,
            num_subjects=num_subjects,
            subject_embedding_dim=subject_embedding_dim,
            expert_embedding_dim=expert_embedding_dim,
            hyper_expert_hidden_dim=hyper_expert_hidden_dim,
            num_channels=num_channels,
            moe_conditioning_dim=moe_conditioning_dim
        )

        self.created_modules.append(module)
        return module
    
    def get_optimized_global_pool(self):
        return self.optimized_global_pool

    def set_subject_ids_for_all_modules(self,subject_ids):
        for module in self.created_modules:
            module.set_subject_ids(subject_ids)

    def get_total_parameter_efficiency(self):
        global_efficiency = self.optimized_global_pool.get_parameter_efficiency_report()
        
        total_local_params = sum(
            sum(p.numel() for p in module.layer_norm.parameters())
            for module in self.created_modules
        )
        
        return {
            'optimized_global_pool_params': global_efficiency['total_global_pool_params'],
            'total_local_module_params': total_local_params,
            'total_system_params': global_efficiency['total_global_pool_params'] + total_local_params,
            'independent_system_params': global_efficiency['total_independent_params'] + total_local_params,
            'overall_parameter_reduction': global_efficiency['parameter_reduction_ratio'],
            'overall_parameter_savings': global_efficiency['parameter_savings'],
            'memory_efficiency_score': global_efficiency['memory_efficiency_score'],
            'num_layers': len(self.created_modules),
            'experts_per_layer': self.num_experts,
            'router_sharing_enabled': True,
            'layer_adapters_enabled': False
        }
    
    def freeze_all_experts(self):
        self.optimized_global_pool.freeze_experts()
        
    def unfreeze_all_experts(self):
        self.optimized_global_pool.unfreeze_experts()
    
    def freeze_shared_router(self):
        self.optimized_global_pool.freeze_router()
        
    def unfreeze_shared_router(self):
        self.optimized_global_pool.unfreeze_router()
        
    def freeze_all_modules(self):
        for module in self.created_modules:
            module.freeze_parameters()
            
    def unfreeze_all_modules(self):
        for module in self.created_modules:
            module.unfreeze_parameters()

    def freeze_all_shared_components(self):
        for module in self.created_modules:
            module.freeze_shared_components()

    def unfreeze_all_shared_components(self):
        for module in self.created_modules:
            module.unfreeze_shared_components()

    def get_comprehensive_analysis(self,x):
        analysis = {
            'factory_stats': self.get_total_parameter_efficiency(),
            'module_analyses': {}
        }

        for i, module in enumerate(self.created_modules):
            module_analysis = {
                'layer_id': module.layer_id,
                'frequency_specialization': module.analyze_frequency_specialization(x),
                'knowledge_representation': module.get_knowledge_representation(x),
                'parameter_efficiency': module.get_parameter_efficiency_metrics(),
                'expert_usage': module.get_expert_stats()
            }
            analysis['module_analyses'][f'layer_{module.layer_id}'] = module_analysis

        return analysis
