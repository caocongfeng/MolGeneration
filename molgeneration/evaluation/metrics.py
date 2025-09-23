"""Evaluation metrics for molecular generation."""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Any
from abc import ABC, abstractmethod
from molgeneration.utils.smiles_utils import SMILESValidator
from molgeneration.utils.chemical_utils import PropertyCalculator, MolecularSimilarity


class BaseMetric(ABC):
    """Base class for evaluation metrics."""
    
    def __init__(self, name: str):
        """Initialize metric."""
        self.name = name
    
    @abstractmethod
    def compute(self, generated_molecules: List[str], reference_molecules: Optional[List[str]] = None) -> float:
        """Compute metric value."""
        pass
    
    def __call__(self, generated_molecules: List[str], reference_molecules: Optional[List[str]] = None) -> float:
        """Call compute method."""
        return self.compute(generated_molecules, reference_molecules)


class ValidityMetric(BaseMetric):
    """Computes the validity of generated molecules."""
    
    def __init__(self):
        """Initialize validity metric."""
        super().__init__("validity")
        self.validator = SMILESValidator()
    
    def compute(self, generated_molecules: List[str], reference_molecules: Optional[List[str]] = None) -> float:
        """
        Compute validity score (fraction of valid SMILES).
        
        Args:
            generated_molecules: List of generated SMILES
            reference_molecules: Not used for validity
            
        Returns:
            Validity score (0-1)
        """
        if not generated_molecules:
            return 0.0
        
        valid_count = sum(1 for mol in generated_molecules if self.validator.is_valid(mol))
        return valid_count / len(generated_molecules)


class UniquenessMetric(BaseMetric):
    """Computes the uniqueness of generated molecules."""
    
    def __init__(self, k: Optional[int] = None):
        """
        Initialize uniqueness metric.
        
        Args:
            k: Consider only top k molecules (if None, use all)
        """
        super().__init__("uniqueness")
        self.k = k
        self.validator = SMILESValidator()
    
    def compute(self, generated_molecules: List[str], reference_molecules: Optional[List[str]] = None) -> float:
        """
        Compute uniqueness score (fraction of unique valid molecules).
        
        Args:
            generated_molecules: List of generated SMILES
            reference_molecules: Not used for uniqueness
            
        Returns:
            Uniqueness score (0-1)
        """
        if not generated_molecules:
            return 0.0
        
        # Consider only top k molecules if specified
        molecules = generated_molecules[:self.k] if self.k else generated_molecules
        
        # Filter valid molecules
        valid_molecules = [mol for mol in molecules if self.validator.is_valid(mol)]
        
        if not valid_molecules:
            return 0.0
        
        # Count unique molecules
        unique_molecules = set(valid_molecules)
        return len(unique_molecules) / len(valid_molecules)


class NoveltyMetric(BaseMetric):
    """Computes the novelty of generated molecules compared to training set."""
    
    def __init__(self):
        """Initialize novelty metric."""
        super().__init__("novelty")
        self.validator = SMILESValidator()
    
    def compute(self, generated_molecules: List[str], reference_molecules: Optional[List[str]] = None) -> float:
        """
        Compute novelty score (fraction of molecules not in reference set).
        
        Args:
            generated_molecules: List of generated SMILES
            reference_molecules: Reference/training SMILES
            
        Returns:
            Novelty score (0-1)
        """
        if not generated_molecules:
            return 0.0
        
        if reference_molecules is None:
            # If no reference provided, all molecules are considered novel
            return 1.0
        
        # Filter valid molecules
        valid_generated = [mol for mol in generated_molecules if self.validator.is_valid(mol)]
        
        if not valid_generated:
            return 0.0
        
        # Convert reference to set for fast lookup
        reference_set = set(reference_molecules)
        
        # Count novel molecules
        novel_count = sum(1 for mol in valid_generated if mol not in reference_set)
        return novel_count / len(valid_generated)


class SimilarityMetric(BaseMetric):
    """Computes similarity metrics between generated and reference molecules."""
    
    def __init__(self, metric_type: str = "tanimoto"):
        """
        Initialize similarity metric.
        
        Args:
            metric_type: Type of similarity metric ('tanimoto', 'dice', 'cosine')
        """
        super().__init__(f"similarity_{metric_type}")
        self.metric_type = metric_type
        self.similarity_calc = MolecularSimilarity(metric=metric_type)
        self.validator = SMILESValidator()
    
    def compute(self, generated_molecules: List[str], reference_molecules: Optional[List[str]] = None) -> float:
        """
        Compute average similarity to reference molecules.
        
        Args:
            generated_molecules: List of generated SMILES
            reference_molecules: Reference SMILES
            
        Returns:
            Average similarity score (0-1)
        """
        if not generated_molecules or not reference_molecules:
            return 0.0
        
        # Filter valid molecules
        valid_generated = [mol for mol in generated_molecules if self.validator.is_valid(mol)]
        valid_reference = [mol for mol in reference_molecules if self.validator.is_valid(mol)]
        
        if not valid_generated or not valid_reference:
            return 0.0
        
        similarities = []
        
        for gen_mol in valid_generated:
            max_similarity = 0.0
            for ref_mol in valid_reference:
                sim = self.similarity_calc.calculate_similarity(gen_mol, ref_mol)
                if sim is not None:
                    max_similarity = max(max_similarity, sim)
            similarities.append(max_similarity)
        
        return np.mean(similarities) if similarities else 0.0


class PropertyMetric(BaseMetric):
    """Computes property-based metrics for generated molecules."""
    
    def __init__(self, property_name: str, target_value: Optional[float] = None, tolerance: float = 0.1):
        """
        Initialize property metric.
        
        Args:
            property_name: Name of molecular property
            target_value: Target property value
            tolerance: Tolerance for target matching
        """
        super().__init__(f"property_{property_name}")
        self.property_name = property_name
        self.target_value = target_value
        self.tolerance = tolerance
        self.property_calc = PropertyCalculator()
        self.validator = SMILESValidator()
    
    def compute(self, generated_molecules: List[str], reference_molecules: Optional[List[str]] = None) -> float:
        """
        Compute property-based score.
        
        Args:
            generated_molecules: List of generated SMILES
            reference_molecules: Not used for property metrics
            
        Returns:
            Property score (depends on specific property)
        """
        if not generated_molecules:
            return 0.0
        
        # Filter valid molecules
        valid_molecules = [mol for mol in generated_molecules if self.validator.is_valid(mol)]
        
        if not valid_molecules:
            return 0.0
        
        property_values = []
        
        for mol in valid_molecules:
            if self.property_name == 'qed':
                value = self.property_calc.calculate_qed(mol)
            else:
                props = self.property_calc.calculate_all_properties(mol)
                value = props.get(self.property_name)
            
            if value is not None:
                property_values.append(value)
        
        if not property_values:
            return 0.0
        
        if self.target_value is not None:
            # Compute fraction within tolerance of target
            within_tolerance = sum(
                1 for val in property_values 
                if abs(val - self.target_value) <= self.tolerance
            )
            return within_tolerance / len(property_values)
        else:
            # Return average property value
            return np.mean(property_values)


class DrugLikenessMetric(BaseMetric):
    """Computes drug-likeness metrics."""
    
    def __init__(self):
        """Initialize drug-likeness metric."""
        super().__init__("drug_likeness")
        self.property_calc = PropertyCalculator()
        self.validator = SMILESValidator()
    
    def compute(self, generated_molecules: List[str], reference_molecules: Optional[List[str]] = None) -> float:
        """
        Compute drug-likeness score (fraction passing Lipinski and Veber rules).
        
        Args:
            generated_molecules: List of generated SMILES
            reference_molecules: Not used
            
        Returns:
            Drug-likeness score (0-1)
        """
        if not generated_molecules:
            return 0.0
        
        # Filter valid molecules
        valid_molecules = [mol for mol in generated_molecules if self.validator.is_valid(mol)]
        
        if not valid_molecules:
            return 0.0
        
        drug_like_count = 0
        
        for mol in valid_molecules:
            props = self.property_calc.calculate_all_properties(mol)
            
            if props.get('drug_like', False):
                drug_like_count += 1
        
        return drug_like_count / len(valid_molecules)


class DiversityMetric(BaseMetric):
    """Computes diversity metrics for generated molecules."""
    
    def __init__(self, metric_type: str = "tanimoto"):
        """
        Initialize diversity metric.
        
        Args:
            metric_type: Type of similarity metric for diversity computation
        """
        super().__init__(f"diversity_{metric_type}")
        self.similarity_calc = MolecularSimilarity(metric=metric_type)
        self.validator = SMILESValidator()
    
    def compute(self, generated_molecules: List[str], reference_molecules: Optional[List[str]] = None) -> float:
        """
        Compute diversity score (1 - average pairwise similarity).
        
        Args:
            generated_molecules: List of generated SMILES
            reference_molecules: Not used
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if not generated_molecules or len(generated_molecules) < 2:
            return 0.0
        
        # Filter valid molecules
        valid_molecules = [mol for mol in generated_molecules if self.validator.is_valid(mol)]
        
        if len(valid_molecules) < 2:
            return 0.0
        
        # Compute pairwise similarities
        similarities = []
        n = len(valid_molecules)
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.similarity_calc.calculate_similarity(
                    valid_molecules[i], valid_molecules[j]
                )
                if sim is not None:
                    similarities.append(sim)
        
        if not similarities:
            return 0.0
        
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity  # Diversity = 1 - similarity


class ScaffoldDiversityMetric(BaseMetric):
    """Computes scaffold diversity of generated molecules."""
    
    def __init__(self):
        """Initialize scaffold diversity metric."""
        super().__init__("scaffold_diversity")
        self.validator = SMILESValidator()
    
    def compute(self, generated_molecules: List[str], reference_molecules: Optional[List[str]] = None) -> float:
        """
        Compute scaffold diversity (fraction of unique scaffolds).
        
        Args:
            generated_molecules: List of generated SMILES
            reference_molecules: Not used
            
        Returns:
            Scaffold diversity score (0-1)
        """
        if not generated_molecules:
            return 0.0
        
        # Filter valid molecules
        valid_molecules = [mol for mol in generated_molecules if self.validator.is_valid(mol)]
        
        if not valid_molecules:
            return 0.0
        
        scaffolds = set()
        
        for mol in valid_molecules:
            scaffold = self._get_scaffold(mol)
            if scaffold:
                scaffolds.add(scaffold)
        
        return len(scaffolds) / len(valid_molecules)
    
    def _get_scaffold(self, smiles: str) -> Optional[str]:
        """Extract molecular scaffold."""
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold:
                return Chem.MolToSmiles(scaffold)
            
            return None
        except:
            return None


class IntDivMetric(BaseMetric):
    """Computes internal diversity metric."""
    
    def __init__(self, k: int = 1000):
        """
        Initialize internal diversity metric.
        
        Args:
            k: Number of molecules to sample for diversity computation
        """
        super().__init__("int_div")
        self.k = k
        self.similarity_calc = MolecularSimilarity()
        self.validator = SMILESValidator()
    
    def compute(self, generated_molecules: List[str], reference_molecules: Optional[List[str]] = None) -> float:
        """
        Compute internal diversity (1 - average pairwise Tanimoto similarity).
        
        Args:
            generated_molecules: List of generated SMILES
            reference_molecules: Not used
            
        Returns:
            Internal diversity score (0-1)
        """
        if not generated_molecules:
            return 0.0
        
        # Filter valid and unique molecules
        valid_molecules = list(set([
            mol for mol in generated_molecules 
            if self.validator.is_valid(mol)
        ]))
        
        if len(valid_molecules) < 2:
            return 0.0
        
        # Sample k molecules if we have more than k
        if len(valid_molecules) > self.k:
            import random
            valid_molecules = random.sample(valid_molecules, self.k)
        
        # Compute pairwise similarities
        similarities = []
        n = len(valid_molecules)
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.similarity_calc.calculate_similarity(
                    valid_molecules[i], valid_molecules[j]
                )
                if sim is not None:
                    similarities.append(sim)
        
        if not similarities:
            return 0.0
        
        return 1.0 - np.mean(similarities)