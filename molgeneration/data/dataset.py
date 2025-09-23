"""Dataset classes for molecular data processing."""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from molgeneration.utils.smiles_utils import (
    SMILESValidator, 
    SMILESCanonicalizer, 
    SMILESTokenizer,
    SMILESAugmenter
)
from molgeneration.utils.chemical_utils import PropertyCalculator


class MolecularDataset(Dataset):
    """Dataset for molecular SMILES and associated data."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[SMILESTokenizer] = None,
        max_length: int = 128,
        canonicalize: bool = True,
        augment: bool = False,
        n_augmentations: int = 5,
        include_properties: bool = False,
        property_names: Optional[List[str]] = None,
        split: str = "train"
    ):
        """
        Initialize molecular dataset.
        
        Args:
            data_path: Path to data file (CSV, JSON, or text)
            tokenizer: SMILES tokenizer
            max_length: Maximum sequence length
            canonicalize: Whether to canonicalize SMILES
            augment: Whether to augment SMILES
            n_augmentations: Number of augmentations per SMILES
            include_properties: Whether to include molecular properties
            property_names: List of property names to include
            split: Dataset split ('train', 'val', 'test')
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer or SMILESTokenizer()
        self.max_length = max_length
        self.canonicalize = canonicalize
        self.augment = augment and split == "train"  # Only augment training data
        self.n_augmentations = n_augmentations
        self.include_properties = include_properties
        self.property_names = property_names or []
        self.split = split
        
        # Initialize utilities
        self.validator = SMILESValidator()
        self.canonicalizer = SMILESCanonicalizer() if canonicalize else None
        self.augmenter = SMILESAugmenter(n_augmentations) if augment else None
        self.property_calc = PropertyCalculator() if include_properties else None
        
        # Load and process data
        self.data = self._load_data()
        self._validate_and_filter_data()
        
        print(f"Loaded {len(self.data)} valid molecules for {split} split")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from file."""
        data = []
        
        if self.data_path.suffix == '.csv':
            df = pd.read_csv(self.data_path)
            for _, row in df.iterrows():
                item = {'smiles': row.get('smiles', row.get('SMILES', ''))}
                
                # Add properties if available
                for prop_name in self.property_names:
                    if prop_name in row:
                        item[prop_name] = row[prop_name]
                
                data.append(item)
        
        elif self.data_path.suffix == '.json':
            with open(self.data_path, 'r') as f:
                data = json.load(f)
        
        elif self.data_path.suffix == '.txt':
            with open(self.data_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append({'smiles': line})
        
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        return data
    
    def _validate_and_filter_data(self) -> None:
        """Validate and filter SMILES data."""
        valid_data = []
        
        for item in self.data:
            smiles = item['smiles']
            
            # Validate SMILES
            if not self.validator.is_valid(smiles):
                continue
            
            # Canonicalize if requested
            if self.canonicalize:
                canonical_smiles = self.canonicalizer.canonicalize(smiles)
                if canonical_smiles is None:
                    continue
                item['smiles'] = canonical_smiles
            
            # Calculate properties if requested
            if self.include_properties and self.property_calc:
                properties = self.property_calc.calculate_all_properties(smiles)
                item.update(properties)
            
            valid_data.append(item)
        
        self.data = valid_data
    
    def __len__(self) -> int:
        """Get dataset length."""
        if self.augment:
            return len(self.data) * (1 + self.n_augmentations)
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item."""
        if self.augment:
            # Determine if this is an augmented sample
            original_idx = idx // (1 + self.n_augmentations)
            augment_idx = idx % (1 + self.n_augmentations)
            
            item = self.data[original_idx].copy()
            smiles = item['smiles']
            
            # Apply augmentation if needed
            if augment_idx > 0:
                augmented_smiles = self.augmenter.augment(smiles)
                if augmented_smiles and len(augmented_smiles) >= augment_idx:
                    smiles = augmented_smiles[augment_idx - 1]
        else:
            item = self.data[idx].copy()
            smiles = item['smiles']
        
        # Tokenize SMILES
        tokens = self.tokenizer.encode(smiles)
        
        # Pad or truncate to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens += [self.tokenizer.token_to_id['<pad>']] * (self.max_length - len(tokens))
        
        result = {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(
                [1 if token != self.tokenizer.token_to_id['<pad>'] else 0 for token in tokens],
                dtype=torch.long
            ),
            'smiles': smiles,
        }
        
        # Add properties if available
        if self.include_properties:
            for prop_name in self.property_names:
                if prop_name in item and item[prop_name] is not None:
                    result[prop_name] = torch.tensor(item[prop_name], dtype=torch.float)
        
        return result
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer.vocab)
    
    def save_tokenizer(self, path: str) -> None:
        """Save tokenizer vocabulary."""
        self.tokenizer.save_vocab(path)


class ChemicalReasoningDataset(Dataset):
    """Dataset for chemical reasoning tasks."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[SMILESTokenizer] = None,
        max_length: int = 512,
        reasoning_templates: Optional[Dict[str, str]] = None,
        split: str = "train"
    ):
        """
        Initialize chemical reasoning dataset.
        
        Args:
            data_path: Path to reasoning data file
            tokenizer: Text tokenizer
            max_length: Maximum sequence length
            reasoning_templates: Templates for reasoning tasks
            split: Dataset split
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # Default reasoning templates
        self.templates = reasoning_templates or {
            'property_prediction': (
                "Given the molecule {smiles}, predict its {property}. "
                "The {property} is {value}."
            ),
            'reaction_prediction': (
                "Given reactants {reactants}, predict the product. "
                "The product is {product}."
            ),
            'retrosynthesis': (
                "To synthesize {target}, the precursors are {precursors}."
            ),
            'functional_group': (
                "The molecule {smiles} contains the following functional groups: {groups}."
            ),
        }
        
        # Load data
        self.data = self._load_reasoning_data()
        print(f"Loaded {len(self.data)} reasoning examples for {split} split")
    
    def _load_reasoning_data(self) -> List[Dict[str, str]]:
        """Load reasoning data from file."""
        data = []
        
        if self.data_path.suffix == '.json':
            with open(self.data_path, 'r') as f:
                raw_data = json.load(f)
            
            for item in raw_data:
                # Generate reasoning text based on template
                reasoning_text = self._generate_reasoning_text(item)
                if reasoning_text:
                    data.append({
                        'text': reasoning_text,
                        'task_type': item.get('task_type', 'property_prediction'),
                        **item
                    })
        
        return data
    
    def _generate_reasoning_text(self, item: Dict[str, Any]) -> Optional[str]:
        """Generate reasoning text from data item."""
        task_type = item.get('task_type', 'property_prediction')
        template = self.templates.get(task_type)
        
        if not template:
            return None
        
        try:
            return template.format(**item)
        except KeyError:
            return None
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item."""
        item = self.data[idx]
        text = item['text']
        
        # For now, use simple character-level tokenization
        # In practice, you would use a proper language model tokenizer
        tokens = [ord(c) for c in text[:self.max_length]]
        
        # Pad sequence
        if len(tokens) < self.max_length:
            tokens += [0] * (self.max_length - len(tokens))
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(
                [1 if token != 0 else 0 for token in tokens],
                dtype=torch.long
            ),
            'text': text,
            'task_type': item['task_type'],
        }


class MolecularGenerationDataset(Dataset):
    """Dataset specifically for molecule generation tasks."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[SMILESTokenizer] = None,
        max_length: int = 128,
        target_properties: Optional[Dict[str, float]] = None,
        property_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        split: str = "train"
    ):
        """
        Initialize molecular generation dataset.
        
        Args:
            data_path: Path to molecule data
            tokenizer: SMILES tokenizer
            max_length: Maximum SMILES length
            target_properties: Target property values for generation
            property_constraints: Property constraints (min, max) for filtering
            split: Dataset split
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer or SMILESTokenizer()
        self.max_length = max_length
        self.target_properties = target_properties or {}
        self.property_constraints = property_constraints or {}
        self.split = split
        
        # Initialize utilities
        self.validator = SMILESValidator()
        self.property_calc = PropertyCalculator()
        
        # Load and process data
        self.data = self._load_and_filter_data()
        print(f"Loaded {len(self.data)} molecules for generation ({split} split)")
    
    def _load_and_filter_data(self) -> List[Dict[str, Any]]:
        """Load and filter data based on property constraints."""
        # Load basic molecular dataset
        base_dataset = MolecularDataset(
            self.data_path,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            include_properties=True,
            split=self.split
        )
        
        filtered_data = []
        
        for item in base_dataset.data:
            # Check property constraints
            if self._meets_constraints(item):
                filtered_data.append(item)
        
        return filtered_data
    
    def _meets_constraints(self, item: Dict[str, Any]) -> bool:
        """Check if molecule meets property constraints."""
        for prop_name, (min_val, max_val) in self.property_constraints.items():
            if prop_name in item:
                value = item[prop_name]
                if value is None or value < min_val or value > max_val:
                    return False
        
        return True
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item."""
        item = self.data[idx]
        smiles = item['smiles']
        
        # Tokenize SMILES
        tokens = self.tokenizer.encode(smiles)
        
        # Create input (without last token) and target (without first token)
        input_tokens = tokens[:-1] if len(tokens) > 1 else tokens
        target_tokens = tokens[1:] if len(tokens) > 1 else tokens
        
        # Pad sequences
        max_len = self.max_length - 1  # Account for shifted sequences
        
        if len(input_tokens) > max_len:
            input_tokens = input_tokens[:max_len]
            target_tokens = target_tokens[:max_len]
        else:
            pad_token = self.tokenizer.token_to_id['<pad>']
            input_tokens += [pad_token] * (max_len - len(input_tokens))
            target_tokens += [pad_token] * (max_len - len(target_tokens))
        
        result = {
            'input_ids': torch.tensor(input_tokens, dtype=torch.long),
            'labels': torch.tensor(target_tokens, dtype=torch.long),
            'smiles': smiles,
        }
        
        # Add property targets if specified
        for prop_name, target_value in self.target_properties.items():
            result[f'target_{prop_name}'] = torch.tensor(target_value, dtype=torch.float)
        
        # Add actual properties
        for prop_name in ['molecular_weight', 'logp', 'tpsa', 'qed']:
            if prop_name in item and item[prop_name] is not None:
                result[prop_name] = torch.tensor(item[prop_name], dtype=torch.float)
        
        return result
    
    def get_property_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for molecular properties in the dataset."""
        stats = {}
        
        for prop_name in ['molecular_weight', 'logp', 'tpsa', 'qed']:
            values = [item[prop_name] for item in self.data 
                     if prop_name in item and item[prop_name] is not None]
            
            if values:
                stats[prop_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return stats