import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.SA_MoE import SA_MoEFactory, SA_MoE

class FeatureCollector:
    def __init__(self):
        self.gamma_params = []
        self.beta_params = []
        self.pre_alignment_features = []
        self.post_alignment_features = []
        self.layer_ids = []

    def clear(self):
        self.gamma_params.clear()
        self.beta_params.clear()
        self.pre_alignment_features.clear()
        self.post_alignment_features.clear()
        self.layer_ids.clear()

    def collect_layer_data(self,layer_id,gamma,beta,pre_features,post_features):
        self.layer_ids.append(layer_id)
        self.gamma_params.append(gamma.clone().detach())
        self.beta_params.append(beta.clone().detach())
        self.pre_alignment_features.append(pre_features.clone().detach())
        self.post_alignment_features.append(post_features.clone().detach())


class EnhancedClassificationOutput:
    def __init__(self,logits,aux_loss,gamma_params=None,beta_params=None,
                 pre_alignment_features=None,post_alignment_features=None,layer_ids=None):
        self.logits = logits
        self.aux_loss = aux_loss
        self.gamma_params = gamma_params or []
        self.beta_params = beta_params or []
        self.pre_alignment_features = pre_alignment_features or []
        self.post_alignment_features = post_alignment_features or []
        self.layer_ids = layer_ids or []


class OptimizedGlobalExplicitDecouplingT5Block(nn.Module):
    def __init__(self,config,layer_id,has_relative_attention_bias=False,
                 optimized_shared_factory=None,shared_config=None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        
        from transformers.models.t5.modeling_t5 import T5LayerSelfAttention, T5LayerFF
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        
        self.layer.append(T5LayerFF(config))
        
        if optimized_shared_factory is not None:
            top_k = shared_config.get('top_k', 2) if shared_config else 2
            aux_loss_weight = shared_config.get('aux_loss_weight', 0.01) if shared_config else 0.01

            enable_hypernetwork = (
                shared_config.get('enable_shared_backbone_hypernetwork', False) or
                shared_config.get('enable_subject_style_normalization', False)
            ) if shared_config else False

            num_subjects = shared_config.get('num_subjects', 50) if shared_config else 50
            subject_embedding_dim = shared_config.get('subject_embedding_dim', 64) if shared_config else 64
            expert_embedding_dim = shared_config.get('expert_embedding_dim', 32) if shared_config else 32
            hyper_expert_hidden_dim = shared_config.get('hyper_expert_hidden_dim', 64) if shared_config else 64
            num_channels = shared_config.get('num_channels', 16) if shared_config else 16

            self.shared_knowledge = optimized_shared_factory.create_module(
                layer_id=layer_id,
                top_k=top_k,
                aux_loss_weight=aux_loss_weight,
                enable_shared_backbone_hypernetwork=enable_hypernetwork,
                num_subjects=num_subjects,
                subject_embedding_dim=subject_embedding_dim,
                expert_embedding_dim=expert_embedding_dim,
                hyper_expert_hidden_dim=hyper_expert_hidden_dim,
                num_channels=num_channels
            )
        else:
            self.shared_knowledge = None
            
        self.adaptive_knowledge = None
            
        self.gating_network = None
            
    def set_training_stage(self,stage):
        if stage == "source_domain":
            if self.shared_knowledge is not None:
                self.shared_knowledge.unfreeze_parameters()
            if self.adaptive_knowledge is not None:
                self.adaptive_knowledge.set_active(False)
                self.adaptive_knowledge.freeze_parameters()
                
        elif stage == "tta":
            if self.shared_knowledge is not None:
                self.shared_knowledge.freeze_parameters()
            if self.adaptive_knowledge is not None:
                self.adaptive_knowledge.set_active(True)
                self.adaptive_knowledge.unfreeze_parameters()
        else:
            raise ValueError(f"Unknown training stage: {stage}")
    
    def forward(self,hidden_states,attention_mask=None,position_bias=None,
                encoder_hidden_states=None,encoder_attention_mask=None,encoder_decoder_position_bias=None,
                layer_head_mask=None,cross_attn_layer_head_mask=None,past_key_value=None,use_cache=False,
                output_attentions=False,return_dict=True,*args,**kwargs):
        self_attn_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            # layer_head_mask=layer_head_mask,
            # past_key_value=past_key_value,
            # use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        attn_output = self_attn_outputs[0]
        outputs = self_attn_outputs[1:]

        if self.shared_knowledge is not None:
            ffn_output = self.layer[1](attn_output)

            current_subject_ids = getattr(self.shared_knowledge, '_current_subject_ids', None)

            shared_transform = self.shared_knowledge(attn_output, subject_ids=current_subject_ids)

            adaptive_transform = None
            if self.adaptive_knowledge is not None and self.adaptive_knowledge.is_active:
                adaptive_transform = self.adaptive_knowledge(attn_output)

            if adaptive_transform is not None:
                final_output = ffn_output + shared_transform + adaptive_transform
            else:
                final_output = shared_transform
        else:
            final_output = self.layer[1](attn_output)

        outputs = (final_output,) + outputs

        return outputs
    
    def get_aux_loss(self):
        aux_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        if self.shared_knowledge is not None:
            aux_loss += self.shared_knowledge.get_aux_loss()
            
        if self.adaptive_knowledge is not None:
            aux_loss += self.adaptive_knowledge.get_aux_loss()
            
        return aux_loss
    
    def clear_aux_losses(self):
        if self.shared_knowledge is not None:
            self.shared_knowledge.clear_aux_losses()
            
        if self.adaptive_knowledge is not None:
            self.adaptive_knowledge.clear_aux_losses()
    
    def get_knowledge_analysis(self,x):
        analysis = {
            'layer_id': self.layer_id,
            'shared_knowledge': {},
            'adaptive_knowledge': {},
            'optimized_pool_efficiency': {}
        }
        
        if self.shared_knowledge is not None:
            analysis['shared_knowledge'] = self.shared_knowledge.get_expert_stats()
            analysis['optimized_pool_efficiency'] = self.shared_knowledge.get_parameter_efficiency_metrics()
            
        if self.adaptive_knowledge is not None:
            analysis['adaptive_knowledge'] = self.adaptive_knowledge.get_expert_stats()
        
        return analysis


class SageStream(nn.Module):
    def __init__(self,config,decoupling_config=None):
        super().__init__()
        self.config = config
        self.decoupling_config = decoupling_config or {}
        
        from .moment import MOMENT
        base_moment = MOMENT(config)
        
        self.patch_embedding = base_moment.patch_embedding
        self.normalizer = base_moment.normalizer
        self.tokenizer = base_moment.tokenizer
        self.mask_generator = base_moment.mask_generator
        self.seq_len = config.seq_len
        self.patch_len = config.patch_len
        
        self._patch_normalizer()
        
        self.encoder = base_moment.encoder
        encoder_config = self.encoder.config
        
        shared_config = self.decoupling_config.get('shared_config', {})
        self.optimized_shared_factory = SA_MoEFactory(
            d_model=encoder_config.d_model,
            d_ff=encoder_config.d_ff,
            num_experts=shared_config.get('num_experts', 4),
            dropout=shared_config.get('dropout', 0.1), 
            freq_learning_mode=shared_config.get('freq_learning_mode', 'adaptive_filter'),
            routing_strategy=shared_config.get('routing_strategy', 'frequency_aware'),
            expert_dim_ratio=shared_config.get('expert_dim_ratio', 1.0),
            max_freq=shared_config.get('max_freq', 40.0),
            sampling_rate=shared_config.get('sampling_rate', 256.0)
        )
        
        
        for i, original_block in enumerate(self.encoder.block):
            has_relative_attention_bias = (i == 0)
            
            decoupling_block = OptimizedGlobalExplicitDecouplingT5Block(
                config=encoder_config,
                layer_id=i,
                has_relative_attention_bias=has_relative_attention_bias,
                optimized_shared_factory=self.optimized_shared_factory,
                shared_config=shared_config
            )
            
            decoupling_block.layer[0].load_state_dict(original_block.layer[0].state_dict())
            decoupling_block.layer[1].load_state_dict(original_block.layer[1].state_dict())
            
            self.encoder.block[i] = decoupling_block
        
        self.head = None
        
        self.current_stage = "source_domain"
    
    def _patch_normalizer(self):
        original_get_statistics = self.normalizer._get_statistics
        
        def patched_get_statistics(x, mask=None):
            if mask is None:
                mask = torch.ones((x.shape[0], x.shape[-1]), device=x.device)
            n_channels = x.shape[1]
            mask = mask.unsqueeze(1).repeat(1, n_channels, 1).bool()
            masked_x = torch.where(mask, x, torch.nan)
            self.normalizer.mean = torch.nanmean(masked_x, dim=-1, keepdim=True).detach()
            
            tensor_mean = masked_x.nanmean(dim=-1, keepdim=True)
            output = (masked_x - tensor_mean).square().nanmean(dim=-1, keepdim=True)
            self.normalizer.stdev = output.sqrt().detach() + self.normalizer.eps
        
        self.normalizer._get_statistics = patched_get_statistics
        
    def set_training_stage(self,stage):
        self.current_stage = stage
        
        for block in self.encoder.block:
            if isinstance(block, OptimizedGlobalExplicitDecouplingT5Block):
                block.set_training_stage(stage)
        
        if stage == "source_domain":
            self.optimized_shared_factory.unfreeze_all_experts()
            self.optimized_shared_factory.unfreeze_shared_router()
            self.optimized_shared_factory.unfreeze_all_modules()
        elif stage == "tta":
            self.optimized_shared_factory.freeze_all_experts()
            self.optimized_shared_factory.freeze_shared_router()
            self.optimized_shared_factory.freeze_all_modules()
    
    def forward(self,x_enc,mask=None,subject_ids=None):
        if mask is not None:
            mask = mask.to(x_enc.device)
        x_enc = self.normalizer(x_enc, mode="norm", mask=mask)
        
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)
        
        x_enc = self.tokenizer(x_enc)
        
        if mask is None:
            actual_seq_len = x_enc.shape[2] * self.patch_len
            mask = torch.ones((x_enc.shape[0], actual_seq_len), device=x_enc.device)
        
        x_enc = self.patch_embedding(x_enc, mask=mask)
        
        batch_size, n_channels, n_patches, d_model = x_enc.shape
        x_enc = x_enc.reshape(batch_size * n_channels, n_patches, d_model)
        
        if mask is not None:
            from momentfm.utils.masking import Masking
            patch_view_mask = Masking.convert_seq_to_patch_view(mask, self.patch_len)
            attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        else:
            attention_mask = None
        
        if subject_ids is not None:
            self.optimized_shared_factory.set_subject_ids_for_all_modules(subject_ids)

        encoder_outputs = self.encoder(inputs_embeds=x_enc, attention_mask=attention_mask)
        
        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = last_hidden_state.reshape(batch_size, n_channels, n_patches, d_model)
        
        return last_hidden_state.mean(dim=1)
    
    def get_total_aux_loss(self):
        total_aux_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for block in self.encoder.block:
            if isinstance(block, OptimizedGlobalExplicitDecouplingT5Block):
                aux_loss = block.get_aux_loss()
                if aux_loss > 0:
                    total_aux_loss = total_aux_loss + aux_loss
        
        return total_aux_loss

    def clear_all_aux_losses(self):
        for block in self.encoder.block:
            if isinstance(block, OptimizedGlobalExplicitDecouplingT5Block):
                block.clear_aux_losses()
    
    def get_comprehensive_analysis(self,x_enc):
        analysis = {
            'model_type': 'SageStream',
            'current_stage': self.current_stage,
            'layer_analysis': {},
            'optimized_global_efficiency': {},
            'total_parameters': {},
            'expert_utilization': {},
            'memory_savings': {}
        }
        
        with torch.no_grad():
            _ = self.forward(x_enc)
        
        for i, block in enumerate(self.encoder.block):
            if isinstance(block, OptimizedGlobalExplicitDecouplingT5Block):
                layer_analysis = block.get_knowledge_analysis(x_enc)
                analysis['layer_analysis'][f'layer_{i}'] = layer_analysis
        
        analysis['optimized_global_efficiency'] = self.optimized_shared_factory.get_total_parameter_efficiency()
        
        analysis['expert_utilization'] = self.optimized_shared_factory.get_optimized_global_pool().get_usage_statistics()
        
        efficiency_report = analysis['optimized_global_efficiency']
        analysis['memory_savings'] = {
            'parameter_reduction_ratio': efficiency_report['overall_parameter_reduction'],
            'parameter_savings': efficiency_report['overall_parameter_savings'],
            'memory_efficiency_score': efficiency_report['memory_efficiency_score'],
            'router_sharing_enabled': efficiency_report['router_sharing_enabled'],
            'layer_adapters_disabled': not efficiency_report['layer_adapters_enabled']
        }
        
        return analysis
    
    def get_parameter_efficiency_summary(self):
        return self.optimized_shared_factory.get_total_parameter_efficiency()
    
    def freeze_pretrained_components(self):
        for param in self.patch_embedding.parameters():
            param.requires_grad = False
        for param in self.normalizer.parameters():
            param.requires_grad = False
        for param in self.tokenizer.parameters():
            param.requires_grad = False

        for block in self.encoder.block:
            if isinstance(block, OptimizedGlobalExplicitDecouplingT5Block):
                for param in block.layer[0].parameters():
                    param.requires_grad = False
                for param in block.layer[1].parameters():
                    param.requires_grad = False
    
    def get_frequency_specialization_analysis(self,x_enc):
        return self.optimized_shared_factory.get_optimized_global_pool().analyze_frequency_specialization(x_enc)


class SageStreamPipeline:
    def __init__(self,model,task_name="classification",num_class=2,freeze_pretrained=True,freeze_head=True,pretrained_head_info=None,reduction="concat"):
        self.model = model
        self.task_name = task_name
        self.num_class = num_class
        self.freeze_head = freeze_head
        self.pretrained_head_info = pretrained_head_info
        self.reduction = reduction

        if freeze_pretrained:
            self.model.freeze_pretrained_components()

        self._init_task_head()

    def _create_classification_head(self,input_seq_len=None):
        from torch import nn

        class StandardClassificationHead(nn.Module):
            def __init__(self,n_channels,d_model,n_classes,head_dropout=0.1,reduction="concat"):
                super().__init__()
                self.dropout = nn.Dropout(head_dropout)
                if reduction == "mean":
                    self.linear = nn.Linear(d_model, n_classes)
                elif reduction == "concat":
                    self.linear = nn.Linear(n_channels * d_model, n_classes)
                else:
                    raise ValueError(f"Reduction method {reduction} not implemented. Only 'mean' and 'concat' are supported.")

            def forward(self,x,input_mask=None):
                x = torch.mean(x, dim=1)
                x = self.dropout(x)
                y = self.linear(x)
                return y

        n_channels = getattr(self.model.config, 'n_channels', 16)
        d_model = self.model.config.d_model

        if self.reduction == "concat":
            pass
        elif self.reduction == "mean":
            pass
        else:
            raise ValueError(f"Unsupported reduction method: {self.reduction}. Choose from 'concat' or 'mean'.")

        return StandardClassificationHead(
            n_channels=n_channels,
            d_model=d_model,
            n_classes=self.num_class,
            head_dropout=0.1,
            reduction=self.reduction
        )

    def _init_task_head(self):
        if self.task_name == "classification":
            if self.pretrained_head_info is not None:
                pretrained_num_classes = self.pretrained_head_info['num_classes']

                if pretrained_num_classes == self.num_class:
                    self.model.head = self._create_classification_head()
                    linear_layer = self.model.head.linear
                    with torch.no_grad():
                        linear_layer.weight.copy_(self.pretrained_head_info['weight'])
                        linear_layer.bias.copy_(self.pretrained_head_info['bias'])

                else:
                    self.model.head = self._create_classification_head()

                    if self.num_class < pretrained_num_classes:
                        linear_layer = self.model.head.linear
                        pretrained_weight = self.pretrained_head_info['weight']
                        pretrained_bias = self.pretrained_head_info['bias']

                        if pretrained_weight.shape[1] == linear_layer.weight.shape[1]:
                            with torch.no_grad():
                                linear_layer.weight.copy_(pretrained_weight[:self.num_class])
                                linear_layer.bias.copy_(pretrained_bias[:self.num_class])
                        else:
                            pass

            elif hasattr(self.model, 'head') and self.model.head is not None:
                existing_head = self.model.head

                if hasattr(existing_head, 'linear'):
                    if existing_head.linear.out_features == self.num_class:
                        pass
                    else:
                        self.model.head = self._create_classification_head()
                else:
                    last_layer = None
                    for module in existing_head.modules():
                        if isinstance(module, nn.Linear):
                            last_layer = module

                    if last_layer is not None and last_layer.out_features == self.num_class:
                        pass
                    else:
                        self.model.head = self._create_classification_head()
            else:
                input_seq_len = getattr(self.model.config, 'input_seq_len', 256)
                self.model.head = self._create_classification_head(input_seq_len)

            if self.freeze_head:
                for param in self.model.head.parameters():
                    param.requires_grad = False
            else:
                for param in self.model.head.parameters():
                    param.requires_grad = True

        else:
            raise ValueError(f"Unsupported task: {self.task_name}")
    
    def set_training_stage(self,stage):
        self.model.set_training_stage(stage)
    
    def classify(self,x_enc,mask=None,subject_ids=None,collect_features=False):
        batch_size, n_channels, seq_len = x_enc.shape

        feature_collector = None
        if collect_features:
            feature_collector = FeatureCollector()
            self.model.optimized_shared_factory.set_feature_collector_for_all_modules(feature_collector)

        if mask is not None:
            mask = mask.to(x_enc.device)
        x_enc_norm = self.model.normalizer(x_enc, mode="norm", mask=mask)

        x_enc_norm = torch.nan_to_num(x_enc_norm, nan=0, posinf=0, neginf=0)

        x_enc_patches = self.model.tokenizer(x_enc_norm)

        if mask is None:
            mask = torch.ones((x_enc_patches.shape[0], seq_len), device=x_enc.device)

        x_enc_embedded = self.model.patch_embedding(x_enc_patches, mask=mask)

        batch_size, n_channels, n_patches, d_model = x_enc_embedded.shape
        x_enc_reshaped = x_enc_embedded.reshape(batch_size * n_channels, n_patches, d_model)

        if mask is not None:
            from momentfm.utils.masking import Masking
            patch_view_mask = Masking.convert_seq_to_patch_view(mask, self.model.patch_len)
            attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        else:
            attention_mask = None

        if subject_ids is not None:
            self.model.optimized_shared_factory.set_subject_ids_for_all_modules(subject_ids)

        if subject_ids is not None:
            self.model.optimized_shared_factory.set_subject_ids_for_all_modules(subject_ids)

        encoder_outputs = self.model.encoder(inputs_embeds=x_enc_reshaped, attention_mask=attention_mask)

        last_hidden_state = encoder_outputs.last_hidden_state
        features = last_hidden_state.reshape(batch_size, n_channels, n_patches, d_model)

        if self.reduction == "mean":
            enc_out = features.mean(dim=1, keepdim=False)
        elif self.reduction == "concat":
            enc_out = features.permute(0, 2, 3, 1).reshape(
                batch_size, n_patches, d_model * n_channels)
        else:
            raise ValueError(f"Unsupported reduction method: {self.reduction}")

        logits = self.model.head(enc_out, input_mask=mask)

        aux_loss = self.model.get_total_aux_loss()

        if collect_features and feature_collector is not None:
            self.model.optimized_shared_factory.set_feature_collector_for_all_modules(None)

            return EnhancedClassificationOutput(
                logits=logits,
                aux_loss=aux_loss,
                gamma_params=feature_collector.gamma_params,
                beta_params=feature_collector.beta_params,
                pre_alignment_features=feature_collector.pre_alignment_features,
                post_alignment_features=feature_collector.post_alignment_features,
                layer_ids=feature_collector.layer_ids
            )
        else:
            class ClassificationOutput:
                def __init__(self,logits,aux_loss):
                    self.logits = logits
                    self.aux_loss = aux_loss

            return ClassificationOutput(logits, aux_loss)
    
    def get_model_analysis(self,x_enc):
        return self.model.get_comprehensive_analysis(x_enc)
    
    def get_efficiency_report(self):
        return self.model.get_parameter_efficiency_summary()
    
    def to(self,device):
        self.model = self.model.to(device)
        return self
    
    def parameters(self):
        return self.model.parameters()
    
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self,state_dict,strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)
    
    def train(self,mode=True):
        self.model.train(mode)
        return self
    
    def eval(self):
        self.model.eval()
        return self
    
    @classmethod
    def from_pretrained(cls,model_path,decoupling_config=None,task_name="classification",num_class=2,model_kwargs=None,**kwargs):
        import json
        import os
        
        if model_kwargs is not None:
            kwargs.update(model_kwargs)
        
        task_name = kwargs.get('task_name', task_name)
        num_class = kwargs.get('num_class', num_class)
        freeze_pretrained = kwargs.get('freeze_pretrained', True)
        freeze_head = kwargs.get('freeze_head', True)
        reduction = kwargs.get('reduction', 'concat')
        
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        if 't5_config' in config_dict:
            t5_config = config_dict['t5_config']
            moment_config = {
                'seq_len': config_dict.get('seq_len', 512),
                'patch_len': config_dict.get('patch_len', 8),
                'patch_stride_len': config_dict.get('patch_stride_len', 8),
                'n_channels': kwargs.get('n_channels', 16),
                'task_name': config_dict.get('task_name', 'reconstruction'),
                'model_name': config_dict.get('model_name', 'MOMENT'),
                'transformer_type': config_dict.get('transformer_type', 'encoder_only'),
                'device': config_dict.get('device', 'cpu'),
                'transformer_backbone': config_dict.get('transformer_backbone', 'google/flan-t5-small'),
                'model_kwargs': config_dict.get('model_kwargs', {}),
                't5_config': t5_config,
                **t5_config
            }
        else:
            moment_config = config_dict
        
        from momentfm.utils.utils import NamespaceWithDefaults
        config = NamespaceWithDefaults(**moment_config)
        
        model = SageStream(
            config=config, 
            decoupling_config=decoupling_config
        )
        
        pretrained_head_info = None

        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            if 'head.linear.weight' in state_dict and 'head.linear.bias' in state_dict:
                pretrained_head_weight = state_dict['head.linear.weight']
                pretrained_head_bias = state_dict['head.linear.bias']
                pretrained_num_classes = pretrained_head_weight.shape[0]

                pretrained_head_info = {
                    'weight': pretrained_head_weight,
                    'bias': pretrained_head_bias,
                    'num_classes': pretrained_num_classes
                }

                if pretrained_num_classes == num_class:
                    pass
                else:
                    state_dict.pop('head.linear.weight', None)
                    state_dict.pop('head.linear.bias', None)

            model.load_state_dict(state_dict, strict=False)

        
        pipeline = cls(
            model=model,
            task_name=task_name,
            num_class=num_class,
            freeze_pretrained=freeze_pretrained,
            freeze_head=freeze_head,
            pretrained_head_info=pretrained_head_info,
            reduction=reduction
        )
        
        return pipeline
