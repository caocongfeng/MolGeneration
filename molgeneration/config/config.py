"""Configuration classes for MolGeneration."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # Base model configuration
    model_name: str = "gpt2"
    model_type: str = "causal_lm"
    vocab_size: int = 50257
    max_length: int = 512
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    dropout: float = 0.1
    
    # Molecular-specific configurations
    chemical_vocab_size: int = 1000
    use_chemical_embeddings: bool = True
    molecular_attention: bool = True
    
    # Generation configuration
    generation_max_length: int = 128
    generation_temperature: float = 1.0
    generation_top_k: int = 50
    generation_top_p: float = 0.95


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # General training
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Two-stage specific
    stage1_epochs: int = 5
    stage2_epochs: int = 5
    stage1_lr: float = 5e-5
    stage2_lr: float = 1e-5
    
    # Reinforcement learning (Stage 2)
    use_rl: bool = True
    rl_reward_weight: float = 1.0
    rl_kl_weight: float = 0.1
    rl_clip_range: float = 0.2
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "linear"
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    logging_steps: int = 100
    save_steps: int = 1000


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Dataset paths
    reasoning_data_path: str = "data/reasoning"
    generation_data_path: str = "data/generation"
    validation_split: float = 0.1
    test_split: float = 0.1
    
    # Data processing
    max_smiles_length: int = 128
    canonicalize_smiles: bool = True
    augment_smiles: bool = True
    use_selfies: bool = False
    
    # Chemical reasoning data
    include_properties: List[str] = field(default_factory=lambda: [
        "molecular_weight", "logp", "tpsa", "qed"
    ])
    reasoning_template: str = "default"


@dataclass
class MolGenerationConfig:
    """Main configuration class combining all components."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment configuration
    experiment_name: str = "mol_generation_experiment"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Hardware configuration
    device: str = "auto"  # auto, cpu, cuda
    num_gpus: int = 1
    mixed_precision: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    @classmethod
    def from_yaml(cls, path: str) -> "MolGenerationConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'device': self.device,
            'num_gpus': self.num_gpus,
            'mixed_precision': self.mixed_precision,
            'seed': self.seed,
            'deterministic': self.deterministic,
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def update(self, **kwargs) -> None:
        """Update configuration with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Handle nested updates
                for attr_name in ['model', 'training', 'data']:
                    attr = getattr(self, attr_name)
                    if hasattr(attr, key):
                        setattr(attr, key, value)
                        break