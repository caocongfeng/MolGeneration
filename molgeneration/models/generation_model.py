"""Generation model for Stage 2 molecular generation fine-tuning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from transformers import GPT2LMHeadModel, GPT2Config
from molgeneration.config.config import ModelConfig
from molgeneration.models.mol_generation_model import MolecularAttention


class GenerationModel(nn.Module):
    """
    Model for Stage 2: Molecular generation fine-tuning.
    
    This model takes the chemical reasoning capabilities from Stage 1
    and fine-tunes them for controlled molecule generation with
    reinforcement learning from molecular property feedback.
    """
    
    def __init__(self, config: ModelConfig, vocab_size: int):
        """
        Initialize generation model.
        
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
            n_positions=config.generation_max_length,
            n_embd=config.hidden_size,
            n_layer=config.num_layers,
            n_head=config.num_attention_heads,
            resid_pdrop=config.dropout,
            attn_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            pad_token_id=0,
            eos_token_id=1,
        )
        
        # Base generation model
        self.generator = GPT2LMHeadModel(self.lm_config)
        
        # Property-guided generation components
        self.property_conditioner = PropertyConditioner(config)
        self.generation_controller = GenerationController(config)
        
        # Reinforcement learning components
        self.value_head = ValueHead(config)
        self.reward_predictor = RewardPredictor(config)
        
        # Property-specific generation heads
        self.property_generators = nn.ModuleDict({
            'molecular_weight': PropertyGenerator(config),
            'logp': PropertyGenerator(config),
            'tpsa': PropertyGenerator(config),
            'qed': PropertyGenerator(config),
        })
        
        # Generation quality estimator
        self.quality_estimator = QualityEstimator(config)
        
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
        target_properties: Optional[Dict[str, torch.Tensor]] = None,
        return_generation_outputs: bool = True,
        compute_rewards: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for generation training.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            target_properties: Target molecular properties
            return_generation_outputs: Whether to return generation-specific outputs
            compute_rewards: Whether to compute rewards for RL
            
        Returns:
            Dictionary of model outputs
        """
        # Base generation forward pass
        gen_outputs = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        outputs = {
            'logits': gen_outputs.logits,
            'hidden_states': gen_outputs.hidden_states[-1],
        }
        
        if labels is not None:
            outputs['generation_loss'] = gen_outputs.loss
        
        if return_generation_outputs:
            # Property conditioning
            if target_properties:
                conditioned_features = self.property_conditioner(
                    outputs['hidden_states'],
                    target_properties,
                    attention_mask
                )
                outputs['conditioned_features'] = conditioned_features
                
                # Property-specific generation
                property_outputs = {}
                for prop_name, target_value in target_properties.items():
                    if prop_name in self.property_generators:
                        prop_gen = self.property_generators[prop_name]
                        prop_outputs[prop_name] = prop_gen(
                            conditioned_features, target_value, attention_mask
                        )
                outputs['property_outputs'] = property_outputs
            
            # Generation control
            controlled_logits = self.generation_controller(
                outputs['logits'],
                outputs['hidden_states'],
                attention_mask
            )
            outputs['controlled_logits'] = controlled_logits
            
            # Value estimation for RL
            values = self.value_head(outputs['hidden_states'], attention_mask)
            outputs['values'] = values
            
            # Quality estimation
            quality_scores = self.quality_estimator(outputs['hidden_states'], attention_mask)
            outputs['quality_scores'] = quality_scores
            
            # Reward prediction if requested
            if compute_rewards:
                predicted_rewards = self.reward_predictor(
                    outputs['hidden_states'], attention_mask
                )
                outputs['predicted_rewards'] = predicted_rewards
        
        return outputs
    
    def generate_molecules(
        self,
        batch_size: int = 1,
        max_length: int = 128,
        target_properties: Optional[Dict[str, float]] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
        device: str = 'cpu',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate molecules with optional property conditioning.
        
        Args:
            batch_size: Number of molecules to generate
            max_length: Maximum sequence length
            target_properties: Target molecular properties
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            num_return_sequences: Number of sequences per input
            device: Device to run on
            
        Returns:
            Dictionary containing generated sequences and metadata
        """
        # Start tokens (assuming 2 is start token)
        input_ids = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)
        
        # Property conditioning
        if target_properties:
            # Convert target properties to tensors
            target_tensors = {}
            for prop_name, value in target_properties.items():
                target_tensors[prop_name] = torch.full(
                    (batch_size,), value, dtype=torch.float, device=device
                )
        else:
            target_tensors = None
        
        generated_sequences = []
        generation_metadata = []
        
        with torch.no_grad():
            for _ in range(num_return_sequences):
                # Generate with property conditioning
                outputs = self.generate_with_control(
                    input_ids=input_ids,
                    max_length=max_length,
                    target_properties=target_tensors,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    **kwargs
                )
                
                generated_sequences.extend(outputs['sequences'])
                generation_metadata.extend(outputs['metadata'])
        
        return {
            'sequences': generated_sequences,
            'metadata': generation_metadata,
            'target_properties': target_properties
        }
    
    def generate_with_control(
        self,
        input_ids: torch.Tensor,
        max_length: int = 128,
        target_properties: Optional[Dict[str, torch.Tensor]] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate molecules with property control and quality monitoring.
        
        Args:
            input_ids: Starting token IDs
            max_length: Maximum generation length
            target_properties: Target property values
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            
        Returns:
            Generated sequences with metadata
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Track generation metadata
        generation_scores = []
        quality_scores = []
        predicted_rewards = []
        
        # Generate step by step
        current_ids = input_ids
        
        for step in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self.forward(
                input_ids=current_ids,
                target_properties=target_properties,
                return_generation_outputs=True,
                compute_rewards=True
            )
            
            # Get controlled logits
            logits = outputs.get('controlled_logits', outputs['logits'])
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k and top-p filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            current_ids = torch.cat([current_ids, next_tokens], dim=-1)
            
            # Store generation metadata
            generation_scores.append(probs.max(dim=-1)[0].cpu())
            
            if 'quality_scores' in outputs:
                quality_scores.append(outputs['quality_scores'].cpu())
            
            if 'predicted_rewards' in outputs:
                predicted_rewards.append(outputs['predicted_rewards'].cpu())
            
            # Check for end tokens
            if (next_tokens == 1).all():  # Assuming 1 is EOS token
                break
        
        return {
            'sequences': current_ids,
            'metadata': {
                'generation_scores': generation_scores,
                'quality_scores': quality_scores,
                'predicted_rewards': predicted_rewards,
            }
        }
    
    def compute_rl_loss(
        self,
        sequences: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        old_log_probs: torch.Tensor,
        clip_range: float = 0.2
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reinforcement learning loss (PPO-style).
        
        Args:
            sequences: Generated sequences
            rewards: Reward values
            values: Value predictions
            old_log_probs: Previous policy log probabilities
            clip_range: PPO clipping range
            
        Returns:
            Dictionary of RL losses
        """
        # Get current policy log probs
        outputs = self.forward(input_ids=sequences, return_generation_outputs=True)
        logits = outputs['logits']
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        current_log_probs = log_probs.gather(-1, sequences.unsqueeze(-1)).squeeze(-1)
        
        # Compute advantages
        advantages = rewards - values
        
        # PPO loss computation
        ratio = torch.exp(current_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, rewards)
        
        # Entropy bonus for exploration
        entropy = -(F.softmax(logits, dim=-1) * log_probs).sum(dim=-1).mean()
        
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        return {
            'total_rl_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'advantages': advantages.mean()
        }


class PropertyConditioner(nn.Module):
    """Conditions generation on target molecular properties."""
    
    def __init__(self, config: ModelConfig):
        """Initialize property conditioner."""
        super().__init__()
        self.config = config
        
        # Property embedding layers
        self.property_embeddings = nn.ModuleDict({
            'molecular_weight': nn.Linear(1, config.hidden_size // 4),
            'logp': nn.Linear(1, config.hidden_size // 4),
            'tpsa': nn.Linear(1, config.hidden_size // 4),
            'qed': nn.Linear(1, config.hidden_size // 4),
        })
        
        # Conditioning fusion layer
        self.fusion_layer = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.fusion_activation = nn.GELU()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        target_properties: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply property conditioning."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute property embeddings
        property_embeds = []
        for prop_name, prop_value in target_properties.items():
            if prop_name in self.property_embeddings:
                embed = self.property_embeddings[prop_name](prop_value.unsqueeze(-1))
                property_embeds.append(embed)
        
        if property_embeds:
            # Concatenate property embeddings
            combined_prop_embed = torch.cat(property_embeds, dim=-1)
            
            # Expand to sequence length
            combined_prop_embed = combined_prop_embed.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Fuse with hidden states
            combined_features = torch.cat([hidden_states, combined_prop_embed], dim=-1)
            conditioned_features = self.fusion_layer(combined_features)
            conditioned_features = self.fusion_activation(conditioned_features)
            
            return conditioned_features
        
        return hidden_states


class GenerationController(nn.Module):
    """Controls generation process for better molecular validity."""
    
    def __init__(self, config: ModelConfig):
        """Initialize generation controller."""
        super().__init__()
        self.config = config
        
        self.control_layer = nn.Linear(config.hidden_size, config.vocab_size)
        self.control_gate = nn.Linear(config.hidden_size, 1)
    
    def forward(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply generation control."""
        # Compute control adjustments
        control_logits = self.control_layer(hidden_states)
        control_gates = torch.sigmoid(self.control_gate(hidden_states))
        
        # Apply gated control
        controlled_logits = logits + control_gates * control_logits
        
        return controlled_logits


class ValueHead(nn.Module):
    """Value head for reinforcement learning."""
    
    def __init__(self, config: ModelConfig):
        """Initialize value head."""
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1)
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute value estimates."""
        # Pool hidden states
        if attention_mask is not None:
            pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = hidden_states.mean(dim=1)
        
        values = self.value_net(pooled).squeeze(-1)
        return values


class RewardPredictor(nn.Module):
    """Predicts rewards for generated molecules."""
    
    def __init__(self, config: ModelConfig):
        """Initialize reward predictor."""
        super().__init__()
        self.reward_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1)
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict rewards."""
        # Pool hidden states
        if attention_mask is not None:
            pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = hidden_states.mean(dim=1)
        
        rewards = self.reward_net(pooled).squeeze(-1)
        return rewards


class PropertyGenerator(nn.Module):
    """Property-specific generation head."""
    
    def __init__(self, config: ModelConfig):
        """Initialize property generator."""
        super().__init__()
        self.property_projection = nn.Linear(config.hidden_size + 1, config.hidden_size)
        self.property_activation = nn.GELU()
    
    def forward(
        self,
        features: torch.Tensor,
        target_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate property-conditioned features."""
        batch_size, seq_len, hidden_size = features.shape
        
        # Expand target value to sequence
        target_expanded = target_value.unsqueeze(1).unsqueeze(2).expand(-1, seq_len, 1)
        
        # Concatenate features with target
        combined = torch.cat([features, target_expanded], dim=-1)
        
        # Project and activate
        output = self.property_projection(combined)
        output = self.property_activation(output)
        
        return output


class QualityEstimator(nn.Module):
    """Estimates quality of generated molecules."""
    
    def __init__(self, config: ModelConfig):
        """Initialize quality estimator."""
        super().__init__()
        self.quality_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()  # Quality score between 0 and 1
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Estimate quality scores."""
        # Pool hidden states
        if attention_mask is not None:
            pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = hidden_states.mean(dim=1)
        
        quality = self.quality_net(pooled).squeeze(-1)
        return quality