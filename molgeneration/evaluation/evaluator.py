"""Main evaluator for molecular generation models."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from molgeneration.evaluation.metrics import (
    ValidityMetric,
    UniquenessMetric,
    NoveltyMetric,
    SimilarityMetric,
    PropertyMetric,
    DrugLikenessMetric,
    DiversityMetric,
    ScaffoldDiversityMetric,
    IntDivMetric,
)
from molgeneration.utils.smiles_utils import SMILESValidator
from molgeneration.utils.chemical_utils import PropertyCalculator


class MolecularEvaluator:
    """
    Comprehensive evaluator for molecular generation models.
    
    Computes various metrics including validity, uniqueness, novelty,
    similarity, diversity, and property-based metrics.
    """
    
    def __init__(
        self,
        reference_molecules: Optional[List[str]] = None,
        target_properties: Optional[Dict[str, float]] = None,
        property_tolerance: float = 0.1,
        include_property_metrics: bool = True,
        include_diversity_metrics: bool = True,
        diversity_k: int = 1000
    ):
        """
        Initialize molecular evaluator.
        
        Args:
            reference_molecules: Reference molecules for novelty/similarity computation
            target_properties: Target property values for property metrics
            property_tolerance: Tolerance for property target matching
            include_property_metrics: Whether to compute property-based metrics
            include_diversity_metrics: Whether to compute diversity metrics
            diversity_k: Number of molecules to use for diversity computation
        """
        self.reference_molecules = reference_molecules
        self.target_properties = target_properties or {}
        self.property_tolerance = property_tolerance
        self.include_property_metrics = include_property_metrics
        self.include_diversity_metrics = include_diversity_metrics
        self.diversity_k = diversity_k
        
        # Initialize metrics
        self.metrics = self._initialize_metrics()
        
        # Initialize utilities
        self.validator = SMILESValidator()
        self.property_calc = PropertyCalculator()
    
    def _initialize_metrics(self) -> Dict[str, Any]:
        """Initialize all evaluation metrics."""
        metrics = {
            # Basic metrics
            'validity': ValidityMetric(),
            'uniqueness': UniquenessMetric(),
            'novelty': NoveltyMetric(),
            
            # Similarity metrics
            'tanimoto_similarity': SimilarityMetric('tanimoto'),
            'dice_similarity': SimilarityMetric('dice'),
            
            # Drug-likeness
            'drug_likeness': DrugLikenessMetric(),
        }
        
        # Property metrics
        if self.include_property_metrics:
            common_properties = ['molecular_weight', 'logp', 'tpsa', 'qed']
            
            for prop in common_properties:
                # Average property metric
                metrics[f'{prop}_avg'] = PropertyMetric(prop)
                
                # Target-based metric if target specified
                if prop in self.target_properties:
                    metrics[f'{prop}_target'] = PropertyMetric(
                        prop, 
                        self.target_properties[prop], 
                        self.property_tolerance
                    )
        
        # Diversity metrics
        if self.include_diversity_metrics:
            metrics.update({
                'diversity': DiversityMetric('tanimoto'),
                'scaffold_diversity': ScaffoldDiversityMetric(),
                'int_div': IntDivMetric(self.diversity_k),
            })
        
        return metrics
    
    def evaluate(
        self,
        generated_molecules: List[str],
        reference_molecules: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        detailed_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of generated molecules.
        
        Args:
            generated_molecules: List of generated SMILES
            reference_molecules: Reference molecules (overrides instance reference)
            save_path: Path to save evaluation results
            detailed_analysis: Whether to include detailed analysis
            
        Returns:
            Dictionary containing all evaluation results
        """
        if not generated_molecules:
            return {'error': 'No molecules provided for evaluation'}
        
        # Use provided reference or instance reference
        ref_molecules = reference_molecules or self.reference_molecules
        
        print(f"Evaluating {len(generated_molecules)} generated molecules...")
        start_time = time.time()
        
        results = {
            'evaluation_info': {
                'num_generated': len(generated_molecules),
                'num_reference': len(ref_molecules) if ref_molecules else 0,
                'evaluation_time': None,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            },
            'metrics': {},
            'detailed_analysis': {} if detailed_analysis else None
        }
        
        # Compute all metrics
        for metric_name, metric in self.metrics.items():
            try:
                score = metric.compute(generated_molecules, ref_molecules)
                results['metrics'][metric_name] = float(score)
                print(f"  {metric_name}: {score:.4f}")
            except Exception as e:
                print(f"  Error computing {metric_name}: {e}")
                results['metrics'][metric_name] = None
        
        # Detailed analysis
        if detailed_analysis:
            results['detailed_analysis'] = self._detailed_analysis(
                generated_molecules, ref_molecules
            )
        
        # Evaluation time
        results['evaluation_info']['evaluation_time'] = time.time() - start_time
        
        # Save results if path provided
        if save_path:
            self._save_results(results, save_path)
        
        return results
    
    def _detailed_analysis(
        self, 
        generated_molecules: List[str], 
        reference_molecules: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform detailed analysis of generated molecules."""
        analysis = {}
        
        # Filter valid molecules
        valid_molecules = [mol for mol in generated_molecules if self.validator.is_valid(mol)]
        
        analysis['valid_molecules'] = {
            'count': len(valid_molecules),
            'fraction': len(valid_molecules) / len(generated_molecules) if generated_molecules else 0,
            'examples': valid_molecules[:10]  # First 10 examples
        }
        
        if not valid_molecules:
            return analysis
        
        # Property analysis
        if self.include_property_metrics:
            analysis['property_analysis'] = self._analyze_properties(valid_molecules)
        
        # Structural analysis
        analysis['structural_analysis'] = self._analyze_structures(valid_molecules)
        
        # Length analysis
        lengths = [len(mol) for mol in valid_molecules]
        analysis['length_analysis'] = {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': np.min(lengths),
            'max': np.max(lengths),
            'median': np.median(lengths)
        }
        
        # Uniqueness analysis
        unique_molecules = list(set(valid_molecules))
        analysis['uniqueness_analysis'] = {
            'unique_count': len(unique_molecules),
            'duplicate_count': len(valid_molecules) - len(unique_molecules),
            'uniqueness_ratio': len(unique_molecules) / len(valid_molecules)
        }
        
        # Novelty analysis (if reference provided)
        if reference_molecules:
            reference_set = set(reference_molecules)
            novel_molecules = [mol for mol in unique_molecules if mol not in reference_set]
            
            analysis['novelty_analysis'] = {
                'novel_count': len(novel_molecules),
                'known_count': len(unique_molecules) - len(novel_molecules),
                'novelty_ratio': len(novel_molecules) / len(unique_molecules),
                'novel_examples': novel_molecules[:10]
            }
        
        return analysis
    
    def _analyze_properties(self, molecules: List[str]) -> Dict[str, Any]:
        """Analyze molecular properties."""
        properties = ['molecular_weight', 'logp', 'tpsa', 'qed']
        property_values = {prop: [] for prop in properties}
        
        for mol in molecules:
            props = self.property_calc.calculate_all_properties(mol)
            for prop in properties:
                value = props.get(prop)
                if value is not None:
                    property_values[prop].append(value)
        
        property_stats = {}
        for prop, values in property_values.items():
            if values:
                property_stats[prop] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'count': len(values)
                }
                
                # Target analysis if target specified
                if prop in self.target_properties:
                    target = self.target_properties[prop]
                    within_tolerance = sum(
                        1 for val in values 
                        if abs(val - target) <= self.property_tolerance
                    )
                    property_stats[prop]['target_value'] = target
                    property_stats[prop]['within_tolerance'] = within_tolerance
                    property_stats[prop]['target_fraction'] = within_tolerance / len(values)
            else:
                property_stats[prop] = None
        
        return property_stats
    
    def _analyze_structures(self, molecules: List[str]) -> Dict[str, Any]:
        """Analyze structural features of molecules."""
        analysis = {
            'atom_counts': {},
            'bond_counts': {},
            'ring_analysis': {},
            'functional_groups': {}
        }
        
        # Common atoms
        common_atoms = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
        atom_counts = {atom: 0 for atom in common_atoms}
        
        # Count atoms in molecules
        for mol in molecules:
            for atom in common_atoms:
                atom_counts[atom] += mol.count(atom)
        
        total_atoms = sum(atom_counts.values())
        if total_atoms > 0:
            analysis['atom_counts'] = {
                atom: count / total_atoms 
                for atom, count in atom_counts.items()
            }
        
        # Ring analysis
        ring_counts = []
        aromatic_counts = []
        
        for mol in molecules:
            ring_counts.append(mol.count('1') + mol.count('2'))  # Simple ring counting
            aromatic_counts.append(sum(1 for c in mol if c.islower()))  # Aromatic atoms
        
        if ring_counts:
            analysis['ring_analysis'] = {
                'avg_rings_per_molecule': np.mean(ring_counts),
                'avg_aromatic_atoms': np.mean(aromatic_counts),
                'molecules_with_rings': sum(1 for count in ring_counts if count > 0),
                'ring_fraction': sum(1 for count in ring_counts if count > 0) / len(ring_counts)
            }
        
        return analysis
    
    def _save_results(self, results: Dict[str, Any], save_path: str) -> None:
        """Save evaluation results to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix == '.json':
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif save_path.suffix == '.csv':
            # Save metrics as CSV
            metrics_df = pd.DataFrame([results['metrics']])
            metrics_df.to_csv(save_path, index=False)
        else:
            # Default to JSON
            with open(save_path.with_suffix('.json'), 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {save_path}")
    
    def compare_models(
        self,
        model_results: Dict[str, List[str]],
        reference_molecules: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models' generated molecules.
        
        Args:
            model_results: Dictionary mapping model names to generated molecules
            reference_molecules: Reference molecules
            save_path: Path to save comparison results
            
        Returns:
            Comparison results
        """
        print(f"Comparing {len(model_results)} models...")
        
        comparison = {
            'models': list(model_results.keys()),
            'comparison_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'individual_results': {},
            'comparison_metrics': {},
        }
        
        # Evaluate each model
        for model_name, molecules in model_results.items():
            print(f"\nEvaluating {model_name}...")
            results = self.evaluate(
                molecules, 
                reference_molecules, 
                detailed_analysis=False
            )
            comparison['individual_results'][model_name] = results
        
        # Compute comparison metrics
        all_metrics = list(self.metrics.keys())
        comparison_data = []
        
        for model_name in model_results.keys():
            model_metrics = comparison['individual_results'][model_name]['metrics']
            row = {'model': model_name}
            row.update(model_metrics)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Ranking for each metric (higher is better for most metrics)
        for metric in all_metrics:
            if metric in comparison_df.columns:
                comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=False)
        
        # Overall ranking (sum of ranks)
        rank_columns = [col for col in comparison_df.columns if col.endswith('_rank')]
        comparison_df['overall_rank'] = comparison_df[rank_columns].sum(axis=1)
        comparison_df['overall_rank'] = comparison_df['overall_rank'].rank()
        
        comparison['comparison_metrics'] = comparison_df.to_dict('records')
        
        # Best model
        best_model_idx = comparison_df['overall_rank'].idxmin()
        comparison['best_model'] = comparison_df.loc[best_model_idx, 'model']
        
        # Summary statistics
        summary_stats = {}
        for metric in all_metrics:
            if metric in comparison_df.columns:
                values = comparison_df[metric].dropna()
                if len(values) > 0:
                    summary_stats[metric] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'best_model': comparison_df.loc[values.idxmax(), 'model']
                    }
        
        comparison['summary_statistics'] = summary_stats
        
        # Save comparison results
        if save_path:
            self._save_results(comparison, save_path)
        
        return comparison
    
    def evaluate_batch(
        self,
        molecule_batches: Dict[str, List[str]],
        reference_molecules: Optional[List[str]] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiple batches of molecules.
        
        Args:
            molecule_batches: Dictionary mapping batch names to molecule lists
            reference_molecules: Reference molecules
            save_dir: Directory to save individual batch results
            
        Returns:
            Batch evaluation results
        """
        print(f"Evaluating {len(molecule_batches)} batches...")
        
        batch_results = {}
        
        for batch_name, molecules in molecule_batches.items():
            print(f"\nEvaluating batch: {batch_name}")
            
            # Individual batch evaluation
            batch_save_path = None
            if save_dir:
                batch_save_path = Path(save_dir) / f"{batch_name}_evaluation.json"
            
            results = self.evaluate(
                molecules,
                reference_molecules,
                save_path=batch_save_path,
                detailed_analysis=True
            )
            
            batch_results[batch_name] = results
        
        return batch_results
    
    def get_metric_summary(self, results: Dict[str, Any]) -> str:
        """Get a formatted summary of evaluation metrics."""
        if 'metrics' not in results:
            return "No metrics available in results"
        
        metrics = results['metrics']
        
        summary_lines = [
            "=== Molecular Generation Evaluation Summary ===",
            f"Generated molecules: {results['evaluation_info']['num_generated']}",
            f"Evaluation time: {results['evaluation_info']['evaluation_time']:.2f}s",
            "",
            "Core Metrics:",
            f"  Validity:    {metrics.get('validity', 'N/A'):.4f}",
            f"  Uniqueness:  {metrics.get('uniqueness', 'N/A'):.4f}",
            f"  Novelty:     {metrics.get('novelty', 'N/A'):.4f}",
            f"  Diversity:   {metrics.get('diversity', 'N/A'):.4f}",
            "",
            "Property Metrics:",
            f"  QED (avg):   {metrics.get('qed_avg', 'N/A'):.4f}",
            f"  Drug-like:   {metrics.get('drug_likeness', 'N/A'):.4f}",
            "",
            "Similarity Metrics:",
            f"  Tanimoto:    {metrics.get('tanimoto_similarity', 'N/A'):.4f}",
            f"  Scaffold Div: {metrics.get('scaffold_diversity', 'N/A'):.4f}",
            "=" * 47
        ]
        
        return "\n".join(summary_lines)