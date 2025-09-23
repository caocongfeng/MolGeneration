"""SMILES processing utilities."""

import re
from typing import List, Optional, Set, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import selfies as sf


class SMILESValidator:
    """Validates SMILES strings using RDKit."""
    
    def __init__(self, strict: bool = True):
        """
        Initialize validator.
        
        Args:
            strict: If True, perform strict validation including aromaticity
        """
        self.strict = strict
    
    def is_valid(self, smiles: str) -> bool:
        """
        Check if a SMILES string is valid.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            if self.strict:
                # Additional checks for aromaticity and sanitization
                Chem.SanitizeMol(mol)
                
            return True
        except:
            return False
    
    def validate_batch(self, smiles_list: List[str]) -> List[bool]:
        """
        Validate a batch of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of boolean validity flags
        """
        return [self.is_valid(smiles) for smiles in smiles_list]
    
    def filter_valid(self, smiles_list: List[str]) -> List[str]:
        """
        Filter out invalid SMILES from a list.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List containing only valid SMILES
        """
        return [smiles for smiles in smiles_list if self.is_valid(smiles)]


class SMILESCanonicalizer:
    """Canonicalizes SMILES strings."""
    
    def __init__(self, isomeric: bool = True, kekulize: bool = False):
        """
        Initialize canonicalizer.
        
        Args:
            isomeric: Whether to include stereochemistry information
            kekulize: Whether to use Kekule form
        """
        self.isomeric = isomeric
        self.kekulize = kekulize
    
    def canonicalize(self, smiles: str) -> Optional[str]:
        """
        Canonicalize a SMILES string.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Canonical SMILES string or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            if self.kekulize:
                Chem.Kekulize(mol)
            
            canonical_smiles = Chem.MolToSmiles(
                mol, 
                isomericSmiles=self.isomeric,
                kekuleSmiles=self.kekulize,
                canonical=True
            )
            
            return canonical_smiles
        except:
            return None
    
    def canonicalize_batch(self, smiles_list: List[str]) -> List[Optional[str]]:
        """
        Canonicalize a batch of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of canonical SMILES (None for invalid ones)
        """
        return [self.canonicalize(smiles) for smiles in smiles_list]


class SMILESAugmenter:
    """Augments SMILES strings through randomization."""
    
    def __init__(self, n_augmentations: int = 5):
        """
        Initialize augmenter.
        
        Args:
            n_augmentations: Number of augmented versions to generate
        """
        self.n_augmentations = n_augmentations
    
    def augment(self, smiles: str) -> List[str]:
        """
        Generate augmented SMILES strings.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            List of augmented SMILES strings
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [smiles]
            
            augmented = []
            for _ in range(self.n_augmentations):
                random_smiles = Chem.MolToSmiles(
                    mol,
                    canonical=False,
                    doRandom=True,
                    isomericSmiles=True
                )
                augmented.append(random_smiles)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_augmented = []
            for smiles in augmented:
                if smiles not in seen:
                    seen.add(smiles)
                    unique_augmented.append(smiles)
            
            return unique_augmented
        except:
            return [smiles]


class SMILESToSELFIES:
    """Converts between SMILES and SELFIES representations."""
    
    def __init__(self):
        """Initialize converter."""
        self.alphabet = None
    
    def smiles_to_selfies(self, smiles: str) -> Optional[str]:
        """
        Convert SMILES to SELFIES.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            SELFIES string or None if conversion fails
        """
        try:
            return sf.encoder(smiles)
        except:
            return None
    
    def selfies_to_smiles(self, selfies: str) -> Optional[str]:
        """
        Convert SELFIES to SMILES.
        
        Args:
            selfies: Input SELFIES string
            
        Returns:
            SMILES string or None if conversion fails
        """
        try:
            return sf.decoder(selfies)
        except:
            return None
    
    def build_alphabet(self, smiles_list: List[str]) -> Set[str]:
        """
        Build SELFIES alphabet from SMILES list.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Set of SELFIES tokens
        """
        alphabet = set()
        
        for smiles in smiles_list:
            selfies = self.smiles_to_selfies(smiles)
            if selfies:
                tokens = sf.split_selfies(selfies)
                alphabet.update(tokens)
        
        self.alphabet = alphabet
        return alphabet
    
    def get_alphabet_size(self) -> int:
        """Get size of SELFIES alphabet."""
        return len(self.alphabet) if self.alphabet else 0


class SMILESTokenizer:
    """Tokenizes SMILES strings for language model input."""
    
    def __init__(self, vocab_file: Optional[str] = None):
        """
        Initialize tokenizer.
        
        Args:
            vocab_file: Path to vocabulary file
        """
        self.vocab = self._build_default_vocab()
        if vocab_file:
            self.load_vocab(vocab_file)
        
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
    
    def _build_default_vocab(self) -> List[str]:
        """Build default SMILES vocabulary."""
        # Common SMILES tokens
        atoms = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']
        brackets = ['[', ']']
        bonds = ['-', '=', '#', ':', '.', '/', '\\']
        rings = [str(i) for i in range(10)]
        charges = ['+', '-']
        aromatic = ['c', 'n', 'o', 's', 'p']
        special = ['(', ')', '@', '@@']
        
        # Special tokens
        special_tokens = ['<pad>', '<unk>', '<start>', '<end>']
        
        vocab = special_tokens + atoms + brackets + bonds + rings + charges + aromatic + special
        return vocab
    
    def tokenize(self, smiles: str) -> List[str]:
        """
        Tokenize SMILES string.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            List of tokens
        """
        # Simple character-level tokenization for SMILES
        # Could be improved with more sophisticated parsing
        tokens = []
        i = 0
        while i < len(smiles):
            # Handle multi-character tokens like 'Cl', 'Br'
            if i < len(smiles) - 1:
                two_char = smiles[i:i+2]
                if two_char in self.token_to_id:
                    tokens.append(two_char)
                    i += 2
                    continue
            
            # Single character token
            char = smiles[i]
            if char in self.token_to_id:
                tokens.append(char)
            else:
                tokens.append('<unk>')
            i += 1
        
        return tokens
    
    def encode(self, smiles: str) -> List[int]:
        """
        Encode SMILES to token IDs.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(smiles)
        return [self.token_to_id.get(token, self.token_to_id['<unk>']) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to SMILES string.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            SMILES string
        """
        tokens = [self.id_to_token.get(token_id, '<unk>') for token_id in token_ids]
        return ''.join(tokens)
    
    def save_vocab(self, path: str) -> None:
        """Save vocabulary to file."""
        with open(path, 'w') as f:
            for token in self.vocab:
                f.write(f"{token}\n")
    
    def load_vocab(self, path: str) -> None:
        """Load vocabulary from file."""
        with open(path, 'r') as f:
            self.vocab = [line.strip() for line in f]
        
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}