"""Data loading utilities for molecular datasets."""

from typing import Dict, Optional, Union, Any
import torch
from torch.utils.data import DataLoader as TorchDataLoader, RandomSampler, SequentialSampler
from molgeneration.data.dataset import (
    MolecularDataset,
    ChemicalReasoningDataset, 
    MolecularGenerationDataset
)
from molgeneration.utils.smiles_utils import SMILESTokenizer


class DataLoader:
    """Data loader factory for molecular datasets."""
    
    def __init__(self, tokenizer: Optional[SMILESTokenizer] = None):
        """
        Initialize data loader factory.
        
        Args:
            tokenizer: SMILES tokenizer to use
        """
        self.tokenizer = tokenizer or SMILESTokenizer()
    
    def create_molecular_dataloader(
        self,
        data_path: str,
        batch_size: int = 32,
        shuffle: bool = True,
        max_length: int = 128,
        num_workers: int = 0,
        split: str = "train",
        **dataset_kwargs
    ) -> TorchDataLoader:
        """
        Create data loader for molecular dataset.
        
        Args:
            data_path: Path to data file
            batch_size: Batch size
            shuffle: Whether to shuffle data
            max_length: Maximum sequence length
            num_workers: Number of worker processes
            split: Dataset split
            **dataset_kwargs: Additional dataset arguments
            
        Returns:
            PyTorch DataLoader
        """
        dataset = MolecularDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=max_length,
            split=split,
            **dataset_kwargs
        )
        
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        
        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=self._molecular_collate_fn,
            pin_memory=torch.cuda.is_available()
        )
    
    def create_reasoning_dataloader(
        self,
        data_path: str,
        batch_size: int = 16,
        shuffle: bool = True,
        max_length: int = 512,
        num_workers: int = 0,
        split: str = "train",
        **dataset_kwargs
    ) -> TorchDataLoader:
        """
        Create data loader for chemical reasoning dataset.
        
        Args:
            data_path: Path to reasoning data file
            batch_size: Batch size
            shuffle: Whether to shuffle data
            max_length: Maximum sequence length
            num_workers: Number of worker processes
            split: Dataset split
            **dataset_kwargs: Additional dataset arguments
            
        Returns:
            PyTorch DataLoader
        """
        dataset = ChemicalReasoningDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=max_length,
            split=split,
            **dataset_kwargs
        )
        
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        
        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=self._reasoning_collate_fn,
            pin_memory=torch.cuda.is_available()
        )
    
    def create_generation_dataloader(
        self,
        data_path: str,
        batch_size: int = 32,
        shuffle: bool = True,
        max_length: int = 128,
        num_workers: int = 0,
        split: str = "train",
        **dataset_kwargs
    ) -> TorchDataLoader:
        """
        Create data loader for molecular generation dataset.
        
        Args:
            data_path: Path to data file
            batch_size: Batch size
            shuffle: Whether to shuffle data
            max_length: Maximum sequence length
            num_workers: Number of worker processes
            split: Dataset split
            **dataset_kwargs: Additional dataset arguments
            
        Returns:
            PyTorch DataLoader
        """
        dataset = MolecularGenerationDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=max_length,
            split=split,
            **dataset_kwargs
        )
        
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        
        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=self._generation_collate_fn,
            pin_memory=torch.cuda.is_available()
        )
    
    def _molecular_collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Collate function for molecular dataset."""
        collated = {}
        
        # Standard fields
        for key in ['input_ids', 'attention_mask']:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch])
        
        # Property fields (optional)
        property_keys = [k for k in batch[0].keys() 
                        if k not in ['input_ids', 'attention_mask', 'smiles']]
        
        for key in property_keys:
            if all(key in item for item in batch):
                collated[key] = torch.stack([item[key] for item in batch])
        
        # SMILES strings
        if 'smiles' in batch[0]:
            collated['smiles'] = [item['smiles'] for item in batch]
        
        return collated
    
    def _reasoning_collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Collate function for reasoning dataset."""
        collated = {}
        
        # Tensor fields
        for key in ['input_ids', 'attention_mask']:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch])
        
        # String fields
        for key in ['text', 'task_type']:
            if key in batch[0]:
                collated[key] = [item[key] for item in batch]
        
        return collated
    
    def _generation_collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Collate function for generation dataset."""
        collated = {}
        
        # Standard fields
        for key in ['input_ids', 'labels']:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch])
        
        # Property fields
        property_keys = [k for k in batch[0].keys() 
                        if k not in ['input_ids', 'labels', 'smiles'] and 
                        isinstance(batch[0][k], torch.Tensor)]
        
        for key in property_keys:
            if all(key in item for item in batch):
                collated[key] = torch.stack([item[key] for item in batch])
        
        # SMILES strings
        if 'smiles' in batch[0]:
            collated['smiles'] = [item['smiles'] for item in batch]
        
        return collated
    
    def get_vocab_size(self) -> int:
        """Get tokenizer vocabulary size."""
        return len(self.tokenizer.vocab)
    
    def save_tokenizer(self, path: str) -> None:
        """Save tokenizer vocabulary."""
        self.tokenizer.save_vocab(path)
    
    def load_tokenizer(self, path: str) -> None:
        """Load tokenizer vocabulary."""
        self.tokenizer.load_vocab(path)