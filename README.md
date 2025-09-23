# MolGeneration

A comprehensive framework for two-stage training of reasoning Large Language Models (LLMs) for molecular generation. This repository implements a novel approach that first pre-trains models on chemical reasoning tasks, then fine-tunes them for controlled molecule generation using reinforcement learning.

## 🧬 Overview

MolGeneration employs a two-stage training paradigm:

1. **Stage 1: Chemical Reasoning Pre-training** - Models learn chemical concepts, properties, and relationships
2. **Stage 2: Molecular Generation Fine-tuning** - Models are fine-tuned for controlled molecule generation with property-based rewards

## ✨ Key Features

- **Two-stage training pipeline** with knowledge transfer between stages
- **Chemical reasoning capabilities** including property prediction, similarity analysis, and functional group identification
- **Reinforcement learning** for property-guided molecule generation
- **Comprehensive evaluation metrics** for validity, novelty, diversity, and drug-likeness
- **Flexible configuration system** with YAML support
- **Molecular property calculation** and chemical descriptor analysis
- **SMILES processing utilities** with validation, canonicalization, and augmentation
- **Multi-task learning** framework for chemical understanding

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/caocongfeng/MolGeneration.git
cd MolGeneration

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from molgeneration.config.config import MolGenerationConfig
from molgeneration.training.two_stage_trainer import TwoStageTrainer
from molgeneration.data.loader import DataLoader
from molgeneration.utils.smiles_utils import SMILESTokenizer

# Setup configuration
config = MolGenerationConfig()
config.experiment_name = "my_mol_generation"
config.training.stage1_epochs = 10
config.training.stage2_epochs = 5

# Setup data and tokenizer
tokenizer = SMILESTokenizer()
data_loader = DataLoader(tokenizer)

# Initialize trainer
trainer = TwoStageTrainer(
    config=config,
    vocab_size=len(tokenizer.vocab),
    device="cuda"
)

# Train the model
results = trainer.train(
    reasoning_data_loader=reasoning_loader,
    generation_data_loader=generation_loader,
    reference_molecules=reference_smiles
)
```

### Running the Example

```bash
cd examples
python train_example.py
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- RDKit
- Transformers
- Additional dependencies listed in `requirements.txt`

## 🏗️ Architecture

### Model Components

1. **MolGenerationModel**: Main model combining reasoning and generation capabilities
2. **ReasoningModel**: Specialized model for Stage 1 chemical reasoning
3. **GenerationModel**: Specialized model for Stage 2 molecular generation with RL
4. **Molecular Attention**: Chemical-aware attention mechanisms
5. **Property Predictors**: Multi-task heads for property prediction

### Training Pipeline

```
Raw SMILES Data
       ↓
   Data Processing
       ↓
Stage 1: Chemical Reasoning Pre-training
   - Property prediction
   - Similarity reasoning  
   - Functional group identification
   - Reaction prediction
       ↓
Knowledge Transfer
       ↓
Stage 2: Molecular Generation Fine-tuning
   - Property-guided generation
   - Reinforcement learning
   - Quality optimization
       ↓
Final Model
```

## 📊 Evaluation Metrics

The framework includes comprehensive evaluation metrics:

- **Validity**: Fraction of chemically valid molecules
- **Uniqueness**: Fraction of unique molecules
- **Novelty**: Fraction of molecules not in training set
- **Diversity**: Internal molecular diversity
- **Drug-likeness**: Lipinski and Veber rule compliance
- **Property metrics**: Target property achievement
- **Similarity metrics**: Tanimoto, Dice, Cosine similarity

## 🔧 Configuration

The system uses a flexible YAML-based configuration:

```yaml
model:
  model_name: "gpt2"
  hidden_size: 768
  num_layers: 12
  num_attention_heads: 12
  use_chemical_embeddings: true

training:
  stage1_epochs: 10
  stage2_epochs: 5
  batch_size: 32
  stage1_lr: 5e-5
  stage2_lr: 1e-5
  use_rl: true

data:
  max_smiles_length: 128
  canonicalize_smiles: true
  augment_smiles: true
```

## 📁 Project Structure

```
MolGeneration/
├── molgeneration/           # Main package
│   ├── config/             # Configuration management
│   ├── data/               # Data processing and loading
│   ├── models/             # Model architectures
│   ├── training/           # Training pipeline
│   ├── evaluation/         # Evaluation metrics
│   └── utils/              # Utility functions
├── examples/               # Usage examples
├── requirements.txt        # Dependencies
├── setup.py               # Package setup
└── README.md              # This file
```

## 🧪 Data Processing

### SMILES Processing

```python
from molgeneration.utils.smiles_utils import SMILESValidator, SMILESCanonicalizer

validator = SMILESValidator()
canonicalizer = SMILESCanonicalizer()

# Validate and canonicalize SMILES
is_valid = validator.is_valid("CCO")
canonical = canonicalizer.canonicalize("CCO")
```

### Dataset Building

```python
from molgeneration.data.preprocessing import DatasetBuilder

builder = DatasetBuilder("./data")
dataset_info = builder.build_complete_dataset(
    raw_smiles_path="molecules.txt",
    create_reasoning_data=True
)
```

## 🎯 Molecular Generation

### Basic Generation

```python
from molgeneration.models.mol_generation_model import MolGenerationModel

model = MolGenerationModel(config.model, vocab_size)
generated = model.generate(
    input_ids=start_tokens,
    max_length=128,
    temperature=0.8,
    top_k=50
)
```

### Property-Guided Generation

```python
from molgeneration.models.generation_model import GenerationModel

gen_model = GenerationModel(config.model, vocab_size)
molecules = gen_model.generate_molecules(
    batch_size=100,
    target_properties={
        'molecular_weight': 300.0,
        'logp': 2.5,
        'qed': 0.7
    }
)
```

## 📈 Evaluation

### Basic Evaluation

```python
from molgeneration.evaluation.evaluator import MolecularEvaluator

evaluator = MolecularEvaluator(
    reference_molecules=training_smiles,
    target_properties={'qed': 0.5}
)

results = evaluator.evaluate(
    generated_molecules=generated_smiles,
    detailed_analysis=True
)

print(evaluator.get_metric_summary(results))
```

### Model Comparison

```python
model_results = {
    "Model_A": molecules_a,
    "Model_B": molecules_b,
    "Baseline": baseline_molecules
}

comparison = evaluator.compare_models(
    model_results=model_results,
    reference_molecules=reference_smiles
)
```

## 🛠️ Customization

### Custom Reward Functions

```python
class CustomRewardCalculator:
    def compute_reward(self, smiles):
        # Implement custom reward logic
        return reward_value
```

### Custom Evaluation Metrics

```python
from molgeneration.evaluation.metrics import BaseMetric

class CustomMetric(BaseMetric):
    def compute(self, generated_molecules, reference_molecules=None):
        # Implement custom metric
        return metric_value
```

## 📚 Examples

The `examples/` directory contains comprehensive examples:

- `train_example.py`: Complete training pipeline example
- Configuration examples
- Data processing examples
- Evaluation examples

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@software{molgeneration2024,
  title={MolGeneration: Two-Stage Training for Reasoning LLMs for Molecule Generation},
  author={MolGeneration Team},
  year={2024},
  url={https://github.com/caocongfeng/MolGeneration}
}
```

## 🔗 Related Work

- [RDKit](https://github.com/rdkit/rdkit) - Cheminformatics toolkit
- [Transformers](https://github.com/huggingface/transformers) - State-of-the-art ML models
- [SELFIES](https://github.com/aspuru-guzik-group/selfies) - Molecular string representation

## 📞 Support

For questions and support, please open an issue on GitHub or contact the maintainers.

---

**MolGeneration** - Advancing molecular design through intelligent reasoning and generation.
