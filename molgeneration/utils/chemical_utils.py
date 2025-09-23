"""Chemical property calculation utilities."""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen, Fragments, Lipinski
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds
import logging

logger = logging.getLogger(__name__)


class MolecularDescriptors:
    """Calculate molecular descriptors and properties."""
    
    def __init__(self):
        """Initialize descriptor calculator."""
        self.descriptor_functions = {
            'molecular_weight': Descriptors.MolWt,
            'logp': Descriptors.MolLogP,
            'tpsa': CalcTPSA,
            'hbd': Descriptors.NumHDonors,
            'hba': Descriptors.NumHAcceptors,
            'rotatable_bonds': CalcNumRotatableBonds,
            'aromatic_rings': Descriptors.NumAromaticRings,
            'heavy_atoms': Descriptors.HeavyAtomCount,
            'formal_charge': Chem.rdmolops.GetFormalCharge,
            'sp3_fraction': Descriptors.FractionCsp3,
        }
    
    def calculate_descriptors(self, smiles: str) -> Dict[str, Optional[float]]:
        """
        Calculate molecular descriptors for a SMILES string.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Dictionary of descriptor values
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {name: None for name in self.descriptor_functions.keys()}
            
            descriptors = {}
            for name, func in self.descriptor_functions.items():
                try:
                    descriptors[name] = func(mol)
                except Exception as e:
                    logger.warning(f"Failed to calculate {name} for {smiles}: {e}")
                    descriptors[name] = None
            
            return descriptors
        
        except Exception as e:
            logger.error(f"Failed to process molecule {smiles}: {e}")
            return {name: None for name in self.descriptor_functions.keys()}
    
    def calculate_batch(self, smiles_list: List[str]) -> List[Dict[str, Optional[float]]]:
        """
        Calculate descriptors for a batch of SMILES.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of descriptor dictionaries
        """
        return [self.calculate_descriptors(smiles) for smiles in smiles_list]


class PropertyCalculator:
    """Calculate drug-like properties and filters."""
    
    def __init__(self):
        """Initialize property calculator."""
        self.descriptor_calc = MolecularDescriptors()
    
    def calculate_qed(self, smiles: str) -> Optional[float]:
        """
        Calculate QED (Quantitative Estimate of Drug-likeness).
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            QED score (0-1) or None if calculation fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return QED.qed(mol)
        except:
            return None
    
    def calculate_sa_score(self, smiles: str) -> Optional[float]:
        """
        Calculate synthetic accessibility score.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            SA score (1-10, lower is better) or None if calculation fails
        """
        # This would require the SAScore module from RDKit contrib
        # For now, return a placeholder
        return None
    
    def lipinski_rule_of_five(self, smiles: str) -> Dict[str, Any]:
        """
        Check Lipinski's Rule of Five.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Dictionary with rule violations and pass/fail status
        """
        descriptors = self.descriptor_calc.calculate_descriptors(smiles)
        
        if any(v is None for v in descriptors.values()):
            return {'pass': False, 'violations': ['calculation_failed']}
        
        violations = []
        
        # Rule of Five criteria
        if descriptors['molecular_weight'] > 500:
            violations.append('molecular_weight_>500')
        
        if descriptors['logp'] > 5:
            violations.append('logp_>5')
        
        if descriptors['hbd'] > 5:
            violations.append('hbd_>5')
        
        if descriptors['hba'] > 10:
            violations.append('hba_>10')
        
        return {
            'pass': len(violations) <= 1,  # Allow one violation
            'violations': violations,
            'violation_count': len(violations)
        }
    
    def veber_rules(self, smiles: str) -> Dict[str, Any]:
        """
        Check Veber rules for oral bioavailability.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Dictionary with rule violations and pass/fail status
        """
        descriptors = self.descriptor_calc.calculate_descriptors(smiles)
        
        if any(v is None for v in descriptors.values()):
            return {'pass': False, 'violations': ['calculation_failed']}
        
        violations = []
        
        # Veber rules criteria
        if descriptors['rotatable_bonds'] > 10:
            violations.append('rotatable_bonds_>10')
        
        if descriptors['tpsa'] > 140:
            violations.append('tpsa_>140')
        
        return {
            'pass': len(violations) == 0,
            'violations': violations,
            'violation_count': len(violations)
        }
    
    def calculate_all_properties(self, smiles: str) -> Dict[str, Any]:
        """
        Calculate comprehensive set of molecular properties.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Dictionary containing all calculated properties
        """
        properties = {}
        
        # Basic descriptors
        properties.update(self.descriptor_calc.calculate_descriptors(smiles))
        
        # Drug-likeness scores
        properties['qed'] = self.calculate_qed(smiles)
        properties['sa_score'] = self.calculate_sa_score(smiles)
        
        # Rule-based filters
        properties['lipinski'] = self.lipinski_rule_of_five(smiles)
        properties['veber'] = self.veber_rules(smiles)
        
        # Derived properties
        properties['drug_like'] = (
            properties['lipinski']['pass'] and 
            properties['veber']['pass']
        )
        
        return properties


class MolecularSimilarity:
    """Calculate molecular similarity metrics."""
    
    def __init__(self, metric: str = 'tanimoto'):
        """
        Initialize similarity calculator.
        
        Args:
            metric: Similarity metric to use ('tanimoto', 'dice', 'cosine')
        """
        self.metric = metric
        
        from rdkit.Chem import DataStructs
        from rdkit import DataStructs as ds
        
        self.similarity_functions = {
            'tanimoto': ds.TanimotoSimilarity,
            'dice': ds.DiceSimilarity,
            'cosine': ds.CosineSimilarity,
        }
    
    def get_fingerprint(self, smiles: str, radius: int = 2, nbits: int = 2048):
        """
        Get molecular fingerprint.
        
        Args:
            smiles: Input SMILES string
            radius: Fingerprint radius
            nbits: Number of bits in fingerprint
            
        Returns:
            RDKit fingerprint object or None
        """
        try:
            from rdkit.Chem import rdMolDescriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=radius, nBits=nbits
            )
            return fp
        except:
            return None
    
    def calculate_similarity(self, smiles1: str, smiles2: str) -> Optional[float]:
        """
        Calculate similarity between two molecules.
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            
        Returns:
            Similarity score (0-1) or None if calculation fails
        """
        fp1 = self.get_fingerprint(smiles1)
        fp2 = self.get_fingerprint(smiles2)
        
        if fp1 is None or fp2 is None:
            return None
        
        similarity_func = self.similarity_functions.get(self.metric)
        if similarity_func is None:
            raise ValueError(f"Unknown similarity metric: {self.metric}")
        
        return similarity_func(fp1, fp2)
    
    def calculate_similarity_matrix(self, smiles_list: List[str]) -> np.ndarray:
        """
        Calculate pairwise similarity matrix.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Similarity matrix as numpy array
        """
        n = len(smiles_list)
        similarity_matrix = np.zeros((n, n))
        
        # Get all fingerprints first
        fingerprints = [self.get_fingerprint(smiles) for smiles in smiles_list]
        
        # Calculate pairwise similarities
        similarity_func = self.similarity_functions.get(self.metric)
        
        for i in range(n):
            for j in range(i, n):
                if fingerprints[i] is not None and fingerprints[j] is not None:
                    sim = similarity_func(fingerprints[i], fingerprints[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
                else:
                    similarity_matrix[i, j] = 0.0
                    similarity_matrix[j, i] = 0.0
        
        return similarity_matrix
    
    def find_most_similar(
        self, 
        query_smiles: str, 
        candidate_smiles: List[str], 
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find most similar molecules to a query.
        
        Args:
            query_smiles: Query SMILES string
            candidate_smiles: List of candidate SMILES
            top_k: Number of top similar molecules to return
            
        Returns:
            List of (SMILES, similarity_score) tuples
        """
        similarities = []
        
        for candidate in candidate_smiles:
            sim = self.calculate_similarity(query_smiles, candidate)
            if sim is not None:
                similarities.append((candidate, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class MolecularFragments:
    """Analyze molecular fragments and substructures."""
    
    def __init__(self):
        """Initialize fragment analyzer."""
        self.fragment_functions = {
            'benzene_rings': Fragments.fr_benzene,
            'aliphatic_rings': Fragments.fr_aliphatic_carboxylic_acid,
            'aromatic_rings': Fragments.fr_Ar_N,
            'nitro_groups': Fragments.fr_nitro,
            'amines': Fragments.fr_NH2,
            'alcohols': Fragments.fr_Al_OH,
            'carbonyls': Fragments.fr_C_O,
            'ethers': Fragments.fr_ether,
            'esters': Fragments.fr_ester,
            'amides': Fragments.fr_amide,
        }
    
    def count_fragments(self, smiles: str) -> Dict[str, Optional[int]]:
        """
        Count molecular fragments.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Dictionary of fragment counts
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {name: None for name in self.fragment_functions.keys()}
            
            fragments = {}
            for name, func in self.fragment_functions.items():
                try:
                    fragments[name] = func(mol)
                except:
                    fragments[name] = None
            
            return fragments
        except:
            return {name: None for name in self.fragment_functions.keys()}
    
    def has_substructure(self, smiles: str, substructure_smarts: str) -> bool:
        """
        Check if molecule contains a specific substructure.
        
        Args:
            smiles: Input SMILES string
            substructure_smarts: SMARTS pattern for substructure
            
        Returns:
            True if substructure is present
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            pattern = Chem.MolFromSmarts(substructure_smarts)
            
            if mol is None or pattern is None:
                return False
            
            return mol.HasSubstructMatch(pattern)
        except:
            return False