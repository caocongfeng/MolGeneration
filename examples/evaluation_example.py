"""Simple evaluation example for generated molecules."""

from molgeneration.evaluation.evaluator import MolecularEvaluator
from molgeneration.evaluation.metrics import *


def main():
    """Run evaluation example."""
    
    print("MolGeneration Evaluation Example")
    print("=" * 40)
    
    # Sample generated molecules (mix of valid and invalid)
    generated_molecules = [
        "CCO",                    # Ethanol - valid
        "CC(C)O",                # Isopropanol - valid  
        "C1CCCCC1",              # Cyclohexane - valid
        "c1ccccc1",              # Benzene - valid
        "CCC(=O)O",              # Propanoic acid - valid
        "CCN(CC)CC",             # Triethylamine - valid
        "COc1ccccc1",            # Anisole - valid
        "CC(C)(C)O",             # tert-Butanol - valid
        "INVALID_SMILES",        # Invalid SMILES
        "CCC[NH3+]",             # Charged molecule - valid
        "C1=CC=CC=C1O",          # Phenol - valid
        "CC(=O)OCC",             # Ethyl acetate - valid
    ]
    
    # Reference molecules (training set example)
    reference_molecules = [
        "CCO",
        "CC(O)CC", 
        "CCCC",
        "C1CCCC1",
        "c1cccnc1",
        "CCC(=O)N",
        "CCOC(=O)C"
    ]
    
    print(f"Generated molecules: {len(generated_molecules)}")
    print(f"Reference molecules: {len(reference_molecules)}")
    
    # 1. Basic Evaluation
    print("\n1. Basic Evaluation")
    print("-" * 20)
    
    evaluator = MolecularEvaluator(
        reference_molecules=reference_molecules,
        include_property_metrics=True,
        include_diversity_metrics=True
    )
    
    results = evaluator.evaluate(
        generated_molecules=generated_molecules,
        detailed_analysis=True
    )
    
    print(evaluator.get_metric_summary(results))
    
    # 2. Property-Targeted Evaluation
    print("\n2. Property-Targeted Evaluation")
    print("-" * 35)
    
    target_evaluator = MolecularEvaluator(
        reference_molecules=reference_molecules,
        target_properties={
            'molecular_weight': 100.0,  # Target MW around 100
            'logp': 1.0,               # Target LogP around 1
            'qed': 0.5                 # Target QED around 0.5
        },
        property_tolerance=0.2,        # 20% tolerance
        include_property_metrics=True
    )
    
    target_results = target_evaluator.evaluate(
        generated_molecules=generated_molecules,
        detailed_analysis=True
    )
    
    print("Property-targeted evaluation:")
    for metric, value in target_results['metrics'].items():
        if 'target' in metric:
            print(f"  {metric}: {value:.4f}")
    
    # 3. Individual Metric Examples
    print("\n3. Individual Metric Examples")
    print("-" * 32)
    
    # Validity
    validity_metric = ValidityMetric()
    validity = validity_metric.compute(generated_molecules)
    print(f"Validity: {validity:.4f}")
    
    # Uniqueness
    uniqueness_metric = UniquenessMetric()
    uniqueness = uniqueness_metric.compute(generated_molecules)
    print(f"Uniqueness: {uniqueness:.4f}")
    
    # Novelty
    novelty_metric = NoveltyMetric()
    novelty = novelty_metric.compute(generated_molecules, reference_molecules)
    print(f"Novelty: {novelty:.4f}")
    
    # Drug-likeness
    drug_metric = DrugLikenessMetric()
    drug_likeness = drug_metric.compute(generated_molecules)
    print(f"Drug-likeness: {drug_likeness:.4f}")
    
    # Diversity
    diversity_metric = DiversityMetric()
    diversity = diversity_metric.compute(generated_molecules)
    print(f"Diversity: {diversity:.4f}")
    
    # 4. Model Comparison Example
    print("\n4. Model Comparison Example")
    print("-" * 29)
    
    # Simulate results from different models
    model_results = {
        "Model_A": generated_molecules[:6],     # First 6 molecules
        "Model_B": generated_molecules[6:],     # Remaining molecules
        "Baseline": ["CC", "CCC", "CCCC", "C1CCC1", "CCO"]  # Simple baseline
    }
    
    comparison = evaluator.compare_models(
        model_results=model_results,
        reference_molecules=reference_molecules
    )
    
    print(f"Best performing model: {comparison['best_model']}")
    print("\nModel rankings:")
    for result in comparison['comparison_metrics']:
        model_name = result['model']
        overall_rank = result.get('overall_rank', 'N/A')
        validity = result.get('validity', 'N/A')
        uniqueness = result.get('uniqueness', 'N/A')
        print(f"  {model_name}: Rank {overall_rank}, Validity {validity:.3f}, Uniqueness {uniqueness:.3f}")
    
    # 5. Detailed Analysis Example
    print("\n5. Detailed Analysis")
    print("-" * 19)
    
    if 'detailed_analysis' in results:
        analysis = results['detailed_analysis']
        
        print(f"Valid molecules: {analysis['valid_molecules']['count']}/{analysis['valid_molecules']['count']}")
        
        if 'property_analysis' in analysis:
            prop_analysis = analysis['property_analysis']
            print("\nProperty statistics:")
            for prop, stats in prop_analysis.items():
                if stats and isinstance(stats, dict):
                    print(f"  {prop}: mean={stats.get('mean', 0):.2f}, std={stats.get('std', 0):.2f}")
        
        if 'length_analysis' in analysis:
            length_stats = analysis['length_analysis']
            print(f"\nMolecule length: mean={length_stats['mean']:.1f}, range=[{length_stats['min']}-{length_stats['max']}]")
    
    print("\nEvaluation example completed!")


if __name__ == "__main__":
    main()