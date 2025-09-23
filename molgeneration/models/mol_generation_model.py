"""Main molecular generation model combining reasoning and generation capabilities."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoConfig,
    GPT2LMHeadModel,
    GPT2Config,
    GPT2Tokenizer
)
from molgeneration.config.config import ModelConfig


class MolGenerationModel(nn.Module):
    """
    Main model for molecular generation with two-stage training capability.
    
    This model can be trained in two stages:
    1. Stage 1: Chemical reasoning pre-training
    2. Stage 2: Molecular generation fine-tuning
    """
    
    def __init__(self, config: ModelConfig, vocab_size: Optional[int] = None):
        """
        Initialize the molecular generation model.
        
        Args:
            config: Model configuration
            vocab_size: Vocabulary size (overrides config if provided)
        """
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size or config.vocab_size
        
        # Create base language model configuration
        self.lm_config = GPT2Config(
            vocab_size=self.vocab_size,
            n_positions=config.max_length,
            n_embd=config.hidden_size,
            n_layer=config.num_layers,
            n_head=config.num_attention_heads,
            resid_pdrop=config.dropout,
            attn_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            pad_token_id=0,  # Assuming 0 is pad token
            eos_token_id=1,  # Assuming 1 is end token
        )
        
        # Base language model
        self.base_model = GPT2LMHeadModel(self.lm_config)
        
        # Chemical-specific embeddings
        if config.use_chemical_embeddings:
            self.chemical_embedding = nn.Embedding(
                config.chemical_vocab_size, 
                config.hidden_size
            )
            self.chemical_projection = nn.Linear(
                config.hidden_size * 2, 
                config.hidden_size
            )
        
        # Molecular attention mechanism
        if config.molecular_attention:
            self.molecular_attention = MolecularAttention(config.hidden_size)
        
        # Property prediction heads (for multi-task learning)
        self.property_heads = nn.ModuleDict({
            'molecular_weight': nn.Linear(config.hidden_size, 1),
            'logp': nn.Linear(config.hidden_size, 1),
            'tpsa': nn.Linear(config.hidden_size, 1),
            'qed': nn.Linear(config.hidden_size, 1),
        })
        
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
        chemical_features: Optional[torch.Tensor] = None,
        return_property_predictions: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for language modeling
            chemical_features: Chemical feature embeddings
            return_property_predictions: Whether to return property predictions
            
        Returns:
            Dictionary containing model outputs
        """
        batch_size, seq_len = input_ids.shape
        
        # Get base model outputs
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        outputs = {
            'logits': base_outputs.logits,
            'hidden_states': base_outputs.hidden_states[-1],
        }
        
        if labels is not None:
            outputs['loss'] = base_outputs.loss
        
        # Apply chemical embeddings if available
        if self.config.use_chemical_embeddings and chemical_features is not None:
            chemical_embeds = self.chemical_embedding(chemical_features)
            combined_embeds = torch.cat([
                outputs['hidden_states'], 
                chemical_embeds
            ], dim=-1)
            outputs['hidden_states'] = self.chemical_projection(combined_embeds)
        
        # Apply molecular attention
        if self.config.molecular_attention:
            outputs['hidden_states'] = self.molecular_attention(
                outputs['hidden_states'], 
                attention_mask
            )
        
        # Property predictions
        if return_property_predictions:
            # Use pooled representation (mean of non-padded tokens)
            if attention_mask is not None:
                pooled_output = (outputs['hidden_states'] * attention_mask.unsqueeze(-1)).sum(dim=1)
                pooled_output = pooled_output / attention_mask.sum(dim=1, keepdim=True)
            else:
                pooled_output = outputs['hidden_states'].mean(dim=1)
            
            property_predictions = {}
            for prop_name, head in self.property_heads.items():
                property_predictions[prop_name] = head(pooled_output).squeeze(-1)
            
            outputs['property_predictions'] = property_predictions
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate sequences using the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to return
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Generated token sequences
        """
        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
    
    def get_embeddings(self) -> nn.Embedding:
        """Get model embeddings."""
        return self.base_model.transformer.wte
    
    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        """Resize token embeddings."""
        self.base_model.resize_token_embeddings(new_vocab_size)
        self.vocab_size = new_vocab_size
    
    def freeze_base_model(self) -> None:
        """Freeze base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_base_model(self) -> None:
        """Unfreeze base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = True


class MolecularAttention(nn.Module):
    """Attention mechanism specialized for molecular representations."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        """
        Initialize molecular attention.
        
        Args:
            hidden_size: Hidden dimension size
            num_heads: Number of attention heads
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(0.1)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
        # Learnable molecular position embeddings
        self.molecular_position_embeddings = nn.Parameter(
            torch.randn(512, hidden_size)  # Max sequence length
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply molecular attention.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            
        Returns:
            Attended hidden states
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Add molecular position embeddings
        position_embeds = self.molecular_position_embeddings[:seq_len].unsqueeze(0)
        hidden_states = hidden_states + position_embeds
        
        # Multi-head attention
        queries = self.query(hidden_states)
        keys = self.key(hidden_states)
        values = self.value(hidden_states)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attention_scores.masked_fill_(attention_mask == 0, float('-inf'))
        
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, values)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )
        
        output = self.output_projection(context)
        
        # Residual connection
        return hidden_states + output


class PropertyPredictor(nn.Module):
    """Neural network for predicting molecular properties."""
    
    def __init__(
        self, 
        input_size: int, 
        hidden_sizes: List[int] = [512, 256], 
        output_size: int = 1,
        dropout: float = 0.1
    ):
        """
        Initialize property predictor.
        
        Args:
            input_size: Input feature size
            hidden_sizes: List of hidden layer sizes
            output_size: Output size
            dropout: Dropout probability
        """
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.layers(x)


def load_pretrained_model(
    model_name_or_path: str,
    config: Optional[ModelConfig] = None,
    vocab_size: Optional[int] = None
) -> MolGenerationModel:
    """
    Load a pretrained molecular generation model.
    
    Args:
        model_name_or_path: Path to model or HuggingFace model name
        config: Model configuration
        vocab_size: Vocabulary size
        
    Returns:
        Loaded model
    """
    if config is None:
        config = ModelConfig()
    
    model = MolGenerationModel(config, vocab_size)
    
    # Load pretrained weights if available
    try:
        state_dict = torch.load(f"{model_name_or_path}/pytorch_model.bin", map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {model_name_or_path}")
    except FileNotFoundError:
        print(f"No pretrained weights found at {model_name_or_path}, using random initialization")
    
    return model