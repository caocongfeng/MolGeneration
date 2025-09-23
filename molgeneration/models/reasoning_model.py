"""Reasoning model for Stage 1 chemical knowledge pre-training."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import GPT2LMHeadModel, GPT2Config
from molgeneration.config.config import ModelConfig
from molgeneration.models.mol_generation_model import MolecularAttention, PropertyPredictor


class ReasoningModel(nn.Module):
    """
    Model for Stage 1: Chemical reasoning pre-training.
    
    This model is trained to understand chemical concepts, properties,
    and relationships before being fine-tuned for molecule generation.
    """
    
    def __init__(self, config: ModelConfig, vocab_size: int):
        """
        Initialize reasoning model.
        
        Args:
            config: Model configuration
            vocab_size: Vocabulary size
        """
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # Language model configuration
        self.lm_config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=config.max_length,
            n_embd=config.hidden_size,
            n_layer=config.num_layers,
            n_head=config.num_attention_heads,
            resid_pdrop=config.dropout,
            attn_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            pad_token_id=0,
            eos_token_id=1,
        )
        
        # Base language model
        self.language_model = GPT2LMHeadModel(self.lm_config)
        
        # Chemical reasoning components
        self.chemical_encoder = ChemicalReasoningEncoder(config)
        self.reasoning_decoder = ReasoningDecoder(config)
        
        # Multi-task heads for different reasoning tasks
        self.task_heads = nn.ModuleDict({
            'property_prediction': PropertyReasoningHead(config),
            'similarity_reasoning': SimilarityReasoningHead(config),
            'functional_group': FunctionalGroupHead(config),
            'reaction_prediction': ReactionReasoningHead(config),
        })
        
        # Task classifier to identify reasoning task type
        self.task_classifier = nn.Linear(config.hidden_size, len(self.task_heads))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        task_type: Optional[str] = None,
        return_reasoning_outputs: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for reasoning training.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            task_type: Type of reasoning task
            return_reasoning_outputs: Whether to return task-specific outputs
            
        Returns:
            Dictionary of model outputs
        """
        # Base language model forward pass
        lm_outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        outputs = {
            'logits': lm_outputs.logits,
            'hidden_states': lm_outputs.hidden_states[-1],
        }
        
        if labels is not None:
            outputs['lm_loss'] = lm_outputs.loss
        
        # Chemical reasoning processing
        if return_reasoning_outputs:
            # Encode chemical understanding
            chemical_features = self.chemical_encoder(
                outputs['hidden_states'], 
                attention_mask
            )
            
            # Decode reasoning
            reasoning_features = self.reasoning_decoder(
                chemical_features, 
                attention_mask
            )
            
            outputs['chemical_features'] = chemical_features
            outputs['reasoning_features'] = reasoning_features
            
            # Task classification
            pooled_features = self._pool_features(reasoning_features, attention_mask)
            task_logits = self.task_classifier(pooled_features)
            outputs['task_logits'] = task_logits
            
            # Task-specific heads
            if task_type and task_type in self.task_heads:
                task_outputs = self.task_heads[task_type](reasoning_features, attention_mask)
                outputs[f'{task_type}_outputs'] = task_outputs
            else:
                # Run all task heads
                task_outputs = {}
                for task_name, task_head in self.task_heads.items():
                    task_outputs[task_name] = task_head(reasoning_features, attention_mask)
                outputs['all_task_outputs'] = task_outputs
        
        return outputs
    
    def _pool_features(
        self, 
        features: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pool features for classification."""
        if attention_mask is not None:
            pooled = (features * attention_mask.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = features.mean(dim=1)
        return pooled
    
    def generate_reasoning(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        **kwargs
    ) -> torch.Tensor:
        """Generate reasoning text."""
        return self.language_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs
        )


class ChemicalReasoningEncoder(nn.Module):
    """Encoder for chemical reasoning understanding."""
    
    def __init__(self, config: ModelConfig):
        """Initialize chemical reasoning encoder."""
        super().__init__()
        self.config = config
        
        # Multi-layer chemical understanding
        self.chemical_layers = nn.ModuleList([
            ChemicalReasoningLayer(config) for _ in range(2)
        ])
        
        # Chemical knowledge attention
        if config.molecular_attention:
            self.molecular_attention = MolecularAttention(
                config.hidden_size, 
                num_heads=config.num_attention_heads // 2
            )
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through chemical encoder."""
        # Apply chemical reasoning layers
        for layer in self.chemical_layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Apply molecular attention
        if hasattr(self, 'molecular_attention'):
            hidden_states = self.molecular_attention(hidden_states, attention_mask)
        
        return hidden_states


class ChemicalReasoningLayer(nn.Module):
    """Single layer for chemical reasoning."""
    
    def __init__(self, config: ModelConfig):
        """Initialize chemical reasoning layer."""
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads // 2,
            dropout=config.dropout
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through reasoning layer."""
        # Self-attention
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        
        # Transpose for nn.MultiheadAttention (expects [seq_len, batch, hidden_size])
        hidden_states = hidden_states.transpose(0, 1)
        
        attn_output, _ = self.attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=attention_mask == 0 if attention_mask is not None else None
        )
        
        # Transpose back
        attn_output = attn_output.transpose(0, 1)
        hidden_states = residual + attn_output
        
        # Feed forward
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = residual + ff_output
        
        return hidden_states


class ReasoningDecoder(nn.Module):
    """Decoder for reasoning output generation."""
    
    def __init__(self, config: ModelConfig):
        """Initialize reasoning decoder."""
        super().__init__()
        self.config = config
        
        self.reasoning_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.reasoning_activation = nn.GELU()
        self.reasoning_dropout = nn.Dropout(config.dropout)
        
        # Reasoning-specific layer normalization
        self.reasoning_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(
        self, 
        chemical_features: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through reasoning decoder."""
        reasoning_features = self.reasoning_projection(chemical_features)
        reasoning_features = self.reasoning_activation(reasoning_features)
        reasoning_features = self.reasoning_dropout(reasoning_features)
        reasoning_features = self.reasoning_norm(reasoning_features)
        
        return reasoning_features


class PropertyReasoningHead(nn.Module):
    """Head for property prediction reasoning."""
    
    def __init__(self, config: ModelConfig):
        """Initialize property reasoning head."""
        super().__init__()
        self.property_predictor = PropertyPredictor(
            config.hidden_size,
            hidden_sizes=[config.hidden_size // 2, config.hidden_size // 4],
            output_size=4  # MW, LogP, TPSA, QED
        )
    
    def forward(
        self, 
        features: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for property reasoning."""
        # Pool features
        if attention_mask is not None:
            pooled_features = (features * attention_mask.unsqueeze(-1)).sum(dim=1)
            pooled_features = pooled_features / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled_features = features.mean(dim=1)
        
        property_preds = self.property_predictor(pooled_features)
        
        return {
            'property_predictions': property_preds,
            'pooled_features': pooled_features
        }


class SimilarityReasoningHead(nn.Module):
    """Head for molecular similarity reasoning."""
    
    def __init__(self, config: ModelConfig):
        """Initialize similarity reasoning head."""
        super().__init__()
        self.similarity_encoder = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.similarity_classifier = nn.Linear(config.hidden_size // 2, 5)  # 5 similarity levels
    
    def forward(
        self, 
        features: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for similarity reasoning."""
        # Pool features
        if attention_mask is not None:
            pooled_features = (features * attention_mask.unsqueeze(-1)).sum(dim=1)
            pooled_features = pooled_features / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled_features = features.mean(dim=1)
        
        similarity_features = self.similarity_encoder(pooled_features)
        similarity_logits = self.similarity_classifier(similarity_features)
        
        return {
            'similarity_logits': similarity_logits,
            'similarity_features': similarity_features
        }


class FunctionalGroupHead(nn.Module):
    """Head for functional group identification."""
    
    def __init__(self, config: ModelConfig):
        """Initialize functional group head."""
        super().__init__()
        # Common functional groups
        self.num_groups = 15
        self.group_classifier = nn.Linear(config.hidden_size, self.num_groups)
    
    def forward(
        self, 
        features: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for functional group identification."""
        # Pool features
        if attention_mask is not None:
            pooled_features = (features * attention_mask.unsqueeze(-1)).sum(dim=1)
            pooled_features = pooled_features / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled_features = features.mean(dim=1)
        
        group_logits = self.group_classifier(pooled_features)
        
        return {
            'functional_group_logits': group_logits,
            'num_groups': self.num_groups
        }


class ReactionReasoningHead(nn.Module):
    """Head for reaction prediction reasoning."""
    
    def __init__(self, config: ModelConfig):
        """Initialize reaction reasoning head."""
        super().__init__()
        self.reaction_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2)
        )
        
        # Reaction type classification
        self.reaction_classifier = nn.Linear(config.hidden_size // 2, 10)  # 10 reaction types
    
    def forward(
        self, 
        features: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for reaction reasoning."""
        # Pool features
        if attention_mask is not None:
            pooled_features = (features * attention_mask.unsqueeze(-1)).sum(dim=1)
            pooled_features = pooled_features / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled_features = features.mean(dim=1)
        
        reaction_features = self.reaction_encoder(pooled_features)
        reaction_logits = self.reaction_classifier(reaction_features)
        
        return {
            'reaction_logits': reaction_logits,
            'reaction_features': reaction_features
        }