"""Data preprocessing utilities for molecular datasets."""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from sklearn.model_selection import train_test_split
from molgeneration.utils.smiles_utils import SMILESValidator, SMILESCanonicalizer
from molgeneration.utils.chemical_utils import PropertyCalculator


class SMILESProcessor:
    """Processes and cleans SMILES datasets."""
    
    def __init__(self, canonicalize: bool = True, validate: bool = True):
        """
        Initialize SMILES processor.
        
        Args:
            canonicalize: Whether to canonicalize SMILES
            validate: Whether to validate SMILES
        """
        self.canonicalize = canonicalize
        self.validate = validate
        
        self.validator = SMILESValidator() if validate else None
        self.canonicalizer = SMILESCanonicalizer() if canonicalize else None
    
    def process_smiles_list(self, smiles_list: List[str]) -> Tuple[List[str], List[int]]:
        """
        Process a list of SMILES strings.
        
        Args:
            smiles_list: List of input SMILES
            
        Returns:
            Tuple of (processed_smiles, valid_indices)
        """
        processed_smiles = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            # Validate SMILES
            if self.validate and not self.validator.is_valid(smiles):
                continue
            
            # Canonicalize SMILES
            if self.canonicalize:
                canonical_smiles = self.canonicalizer.canonicalize(smiles)
                if canonical_smiles is None:
                    continue
                smiles = canonical_smiles
            
            processed_smiles.append(smiles)
            valid_indices.append(i)
        
        return processed_smiles, valid_indices
    
    def process_csv_file(
        self,
        input_path: str,
        output_path: str,
        smiles_column: str = 'smiles',
        remove_duplicates: bool = True
    ) -> Dict[str, int]:
        """
        Process SMILES from CSV file.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to output CSV file
            smiles_column: Name of SMILES column
            remove_duplicates: Whether to remove duplicate SMILES
            
        Returns:
            Dictionary with processing statistics
        """
        # Load data
        df = pd.read_csv(input_path)
        original_count = len(df)
        
        if smiles_column not in df.columns:
            raise ValueError(f"Column '{smiles_column}' not found in CSV")
        
        # Process SMILES
        smiles_list = df[smiles_column].tolist()
        processed_smiles, valid_indices = self.process_smiles_list(smiles_list)
        
        # Filter dataframe to valid entries
        df_filtered = df.iloc[valid_indices].copy()
        df_filtered[smiles_column] = processed_smiles
        
        # Remove duplicates if requested
        if remove_duplicates:
            df_filtered = df_filtered.drop_duplicates(subset=[smiles_column])
        
        # Save processed data
        df_filtered.to_csv(output_path, index=False)
        
        stats = {
            'original_count': original_count,
            'valid_count': len(processed_smiles),
            'final_count': len(df_filtered),
            'invalid_count': original_count - len(processed_smiles),
            'duplicate_count': len(processed_smiles) - len(df_filtered) if remove_duplicates else 0,
        }
        
        return stats
    
    def create_train_val_test_splits(
        self,
        data_path: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify_column: Optional[str] = None,
        random_state: int = 42
    ) -> Dict[str, str]:
        """
        Create train/validation/test splits from processed data.
        
        Args:
            data_path: Path to processed data file
            output_dir: Directory to save splits
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set  
            test_ratio: Fraction for test set
            stratify_column: Column to stratify splits on
            random_state: Random seed
            
        Returns:
            Dictionary with paths to split files
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Stratification data
        stratify_data = None
        if stratify_column and stratify_column in df.columns:
            stratify_data = df[stratify_column]
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=stratify_data
        )
        
        # Second split: val vs test
        if val_ratio > 0 and test_ratio > 0:
            val_test_ratio = val_ratio / (val_ratio + test_ratio)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1 - val_test_ratio),
                random_state=random_state + 1
            )
        elif val_ratio > 0:
            val_df = temp_df
            test_df = pd.DataFrame()
        else:
            val_df = pd.DataFrame()
            test_df = temp_df
        
        # Save splits
        split_paths = {}
        
        train_path = output_dir / "train.csv"
        train_df.to_csv(train_path, index=False)
        split_paths['train'] = str(train_path)
        
        if len(val_df) > 0:
            val_path = output_dir / "val.csv"
            val_df.to_csv(val_path, index=False)
            split_paths['val'] = str(val_path)
        
        if len(test_df) > 0:
            test_path = output_dir / "test.csv"
            test_df.to_csv(test_path, index=False)
            split_paths['test'] = str(test_path)
        
        # Save split info
        split_info = {
            'train_count': len(train_df),
            'val_count': len(val_df),
            'test_count': len(test_df),
            'total_count': len(df),
            'split_ratios': {
                'train': train_ratio,
                'val': val_ratio,
                'test': test_ratio
            }
        }
        
        info_path = output_dir / "split_info.json"
        with open(info_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        return split_paths


class ChemicalReasoningProcessor:
    """Processes data for chemical reasoning tasks."""
    
    def __init__(self):
        """Initialize reasoning processor."""
        self.property_calc = PropertyCalculator()
    
    def create_property_prediction_data(
        self,
        smiles_data: List[str],
        output_path: str,
        properties: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create property prediction reasoning data.
        
        Args:
            smiles_data: List of SMILES strings
            output_path: Path to save reasoning data
            properties: List of properties to predict
            
        Returns:
            Dictionary with processing statistics
        """
        if properties is None:
            properties = ['molecular_weight', 'logp', 'tpsa', 'qed']
        
        reasoning_data = []
        successful_count = 0
        
        for smiles in smiles_data:
            # Calculate properties
            props = self.property_calc.calculate_all_properties(smiles)
            
            if all(props.get(prop) is not None for prop in properties):
                # Create reasoning examples for each property
                for prop in properties:
                    value = props[prop]
                    
                    reasoning_example = {
                        'task_type': 'property_prediction',
                        'smiles': smiles,
                        'property': prop,
                        'value': value,
                        'input_text': f"Predict the {prop} of molecule {smiles}.",
                        'output_text': f"The {prop} of {smiles} is {value:.3f}."
                    }
                    
                    reasoning_data.append(reasoning_example)
                
                successful_count += 1
        
        # Save reasoning data
        with open(output_path, 'w') as f:
            json.dump(reasoning_data, f, indent=2)
        
        stats = {
            'total_molecules': len(smiles_data),
            'successful_molecules': successful_count,
            'total_examples': len(reasoning_data),
            'properties': properties
        }
        
        return stats
    
    def create_functional_group_data(
        self,
        smiles_data: List[str],
        output_path: str
    ) -> Dict[str, Any]:
        """
        Create functional group identification reasoning data.
        
        Args:
            smiles_data: List of SMILES strings
            output_path: Path to save reasoning data
            
        Returns:
            Dictionary with processing statistics
        """
        reasoning_data = []
        
        # Common functional group patterns (SMARTS)
        functional_groups = {
            'alcohol': '[OH]',
            'amine': '[NX3;H2,H1;!$(NC=O)]',
            'carboxylic_acid': '[CX3](=O)[OX2H1]',
            'ester': '[CX3](=O)[OX2H0]',
            'ether': '[OD2]([#6])[#6]',
            'ketone': '[CX3]=[OX1]',
            'aldehyde': '[CX3H1](=O)[#6]',
            'amide': '[CX3](=[OX1])[NX3]',
            'benzene': 'c1ccccc1',
            'pyridine': 'c1ccncc1',
        }
        
        from molgeneration.utils.chemical_utils import MolecularFragments
        fragment_analyzer = MolecularFragments()
        
        for smiles in smiles_data:
            # Identify functional groups
            present_groups = []
            
            for group_name, smarts in functional_groups.items():
                if fragment_analyzer.has_substructure(smiles, smarts):
                    present_groups.append(group_name)
            
            if present_groups:
                groups_text = ", ".join(present_groups)
                
                reasoning_example = {
                    'task_type': 'functional_group',
                    'smiles': smiles,
                    'groups': groups_text,
                    'input_text': f"Identify functional groups in molecule {smiles}.",
                    'output_text': f"The molecule {smiles} contains: {groups_text}."
                }
                
                reasoning_data.append(reasoning_example)
        
        # Save reasoning data
        with open(output_path, 'w') as f:
            json.dump(reasoning_data, f, indent=2)
        
        stats = {
            'total_molecules': len(smiles_data),
            'molecules_with_groups': len(reasoning_data),
            'functional_groups': list(functional_groups.keys())
        }
        
        return stats
    
    def create_similarity_reasoning_data(
        self,
        smiles_data: List[str],
        output_path: str,
        n_pairs: int = 1000
    ) -> Dict[str, Any]:
        """
        Create molecular similarity reasoning data.
        
        Args:
            smiles_data: List of SMILES strings
            output_path: Path to save reasoning data
            n_pairs: Number of molecule pairs to generate
            
        Returns:
            Dictionary with processing statistics
        """
        from molgeneration.utils.chemical_utils import MolecularSimilarity
        
        similarity_calc = MolecularSimilarity()
        reasoning_data = []
        
        # Generate random pairs
        import random
        pairs = []
        for _ in range(n_pairs):
            mol1, mol2 = random.sample(smiles_data, 2)
            pairs.append((mol1, mol2))
        
        for mol1, mol2 in pairs:
            similarity = similarity_calc.calculate_similarity(mol1, mol2)
            
            if similarity is not None:
                # Categorize similarity
                if similarity > 0.8:
                    similarity_desc = "very similar"
                elif similarity > 0.6:
                    similarity_desc = "similar"
                elif similarity > 0.4:
                    similarity_desc = "moderately similar"
                else:
                    similarity_desc = "dissimilar"
                
                reasoning_example = {
                    'task_type': 'similarity',
                    'molecule1': mol1,
                    'molecule2': mol2,
                    'similarity': similarity,
                    'similarity_desc': similarity_desc,
                    'input_text': f"Compare molecules {mol1} and {mol2}.",
                    'output_text': f"Molecules {mol1} and {mol2} are {similarity_desc} (similarity: {similarity:.3f})."
                }
                
                reasoning_data.append(reasoning_example)
        
        # Save reasoning data
        with open(output_path, 'w') as f:
            json.dump(reasoning_data, f, indent=2)
        
        stats = {
            'requested_pairs': n_pairs,
            'successful_pairs': len(reasoning_data),
            'average_similarity': np.mean([item['similarity'] for item in reasoning_data])
        }
        
        return stats


class DatasetBuilder:
    """Builds complete datasets for molecular generation training."""
    
    def __init__(self, output_dir: str):
        """
        Initialize dataset builder.
        
        Args:
            output_dir: Directory to save built datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.smiles_processor = SMILESProcessor()
        self.reasoning_processor = ChemicalReasoningProcessor()
    
    def build_complete_dataset(
        self,
        raw_smiles_path: str,
        dataset_name: str = "mol_generation_dataset",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        create_reasoning_data: bool = True,
        n_reasoning_examples: int = 10000
    ) -> Dict[str, Any]:
        """
        Build complete dataset from raw SMILES data.
        
        Args:
            raw_smiles_path: Path to raw SMILES file
            dataset_name: Name for the dataset
            train_ratio: Training split ratio
            val_ratio: Validation split ratio  
            test_ratio: Test split ratio
            create_reasoning_data: Whether to create reasoning data
            n_reasoning_examples: Number of reasoning examples to create
            
        Returns:
            Dictionary with dataset information and paths
        """
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Step 1: Process raw SMILES data
        print("Processing SMILES data...")
        processed_smiles_path = dataset_dir / "processed_smiles.csv"
        
        # Load and process SMILES
        if raw_smiles_path.endswith('.csv'):
            processing_stats = self.smiles_processor.process_csv_file(
                raw_smiles_path,
                processed_smiles_path
            )
        else:
            # Handle text files
            with open(raw_smiles_path, 'r') as f:
                raw_smiles = [line.strip() for line in f if line.strip()]
            
            processed_smiles, _ = self.smiles_processor.process_smiles_list(raw_smiles)
            
            # Save as CSV
            df = pd.DataFrame({'smiles': processed_smiles})
            df.to_csv(processed_smiles_path, index=False)
            
            processing_stats = {
                'original_count': len(raw_smiles),
                'final_count': len(processed_smiles),
                'invalid_count': len(raw_smiles) - len(processed_smiles)
            }
        
        # Step 2: Create train/val/test splits
        print("Creating data splits...")
        split_dir = dataset_dir / "splits"
        split_paths = self.smiles_processor.create_train_val_test_splits(
            processed_smiles_path,
            split_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        # Step 3: Create reasoning data
        reasoning_stats = {}
        if create_reasoning_data:
            print("Creating reasoning data...")
            reasoning_dir = dataset_dir / "reasoning"
            reasoning_dir.mkdir(exist_ok=True)
            
            # Use training SMILES for reasoning data
            train_df = pd.read_csv(split_paths['train'])
            train_smiles = train_df['smiles'].tolist()
            
            # Limit number of molecules for reasoning data
            if len(train_smiles) > n_reasoning_examples:
                train_smiles = train_smiles[:n_reasoning_examples]
            
            # Property prediction reasoning
            prop_reasoning_path = reasoning_dir / "property_prediction.json"
            prop_stats = self.reasoning_processor.create_property_prediction_data(
                train_smiles, prop_reasoning_path
            )
            
            # Functional group reasoning
            func_reasoning_path = reasoning_dir / "functional_groups.json"
            func_stats = self.reasoning_processor.create_functional_group_data(
                train_smiles, func_reasoning_path
            )
            
            # Similarity reasoning
            sim_reasoning_path = reasoning_dir / "similarity.json"
            sim_stats = self.reasoning_processor.create_similarity_reasoning_data(
                train_smiles, sim_reasoning_path, n_pairs=min(1000, len(train_smiles) // 2)
            )
            
            reasoning_stats = {
                'property_prediction': prop_stats,
                'functional_groups': func_stats,
                'similarity': sim_stats
            }
        
        # Step 4: Save dataset metadata
        metadata = {
            'dataset_name': dataset_name,
            'creation_date': pd.Timestamp.now().isoformat(),
            'processing_stats': processing_stats,
            'split_paths': split_paths,
            'reasoning_stats': reasoning_stats,
            'total_molecules': processing_stats['final_count'],
            'splits': {
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'test_ratio': test_ratio
            }
        }
        
        metadata_path = dataset_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Dataset built successfully: {dataset_dir}")
        print(f"Total molecules: {processing_stats['final_count']}")
        print(f"Training molecules: {metadata['reasoning_stats'].get('property_prediction', {}).get('total_molecules', 0) if create_reasoning_data else 'N/A'}")
        
        return metadata