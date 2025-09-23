"""Data processing example showing how to prepare molecular datasets."""

from pathlib import Path
from molgeneration.data.preprocessing import DatasetBuilder, SMILESProcessor
from molgeneration.utils.smiles_utils import SMILESValidator, SMILESCanonicalizer
from molgeneration.utils.chemical_utils import PropertyCalculator


def main():
    """Demonstrate data processing capabilities."""
    
    print("MolGeneration Data Processing Example")
    print("=" * 42)
    
    # 1. Create sample molecular data
    print("\n1. Creating Sample Data")
    print("-" * 23)
    
    # Sample SMILES strings (mix of valid and invalid)
    sample_smiles = [
        "CCO",                    # Ethanol
        "CC(C)O",                # Isopropanol
        "C1CCCCC1",              # Cyclohexane
        "c1ccccc1",              # Benzene
        "CCC(=O)O",              # Propanoic acid
        "CCN(CC)CC",             # Triethylamine
        "COc1ccccc1",            # Anisole
        "CC(C)(C)O",             # tert-Butanol
        "INVALID_SMILES",        # Invalid SMILES (will be filtered)
        "CCC[NH3+]",             # Charged molecule
        "C1=CC=CC=C1O",          # Phenol
        "CC(=O)OCC",             # Ethyl acetate
        "c1ccc2ccccc2c1",        # Naphthalene
        "CCc1ccccc1",            # Ethylbenzene
        "CCOC(=O)C",             # Ethyl acetate (different representation)
        "CC(C)CC(C)(C)C",        # 2,2,4-Trimethylpentane
        "C1CCNCC1",              # Piperidine
        "c1cncc(c1)C",           # 4-Methylpyridine
        "CC(=O)NC",              # N-Methylacetamide
        "CCCCCCO",               # 1-Heptanol
    ]
    
    print(f"Created {len(sample_smiles)} sample molecules")
    
    # 2. Basic SMILES Processing
    print("\n2. Basic SMILES Processing")
    print("-" * 27)
    
    # Initialize processors
    validator = SMILESValidator()
    canonicalizer = SMILESCanonicalizer()
    smiles_processor = SMILESProcessor(canonicalize=True, validate=True)
    
    # Process SMILES
    processed_smiles, valid_indices = smiles_processor.process_smiles_list(sample_smiles)
    
    print(f"Input molecules: {len(sample_smiles)}")
    print(f"Valid molecules: {len(processed_smiles)}")
    print(f"Invalid molecules: {len(sample_smiles) - len(processed_smiles)}")
    
    # Show examples
    print("\nProcessing examples:")
    for i, smiles in enumerate(sample_smiles[:5]):
        is_valid = validator.is_valid(smiles)
        canonical = canonicalizer.canonicalize(smiles) if is_valid else None
        print(f"  {smiles} -> Valid: {is_valid}, Canonical: {canonical}")
    
    # 3. Property Calculation
    print("\n3. Molecular Property Calculation")
    print("-" * 35)
    
    property_calc = PropertyCalculator()
    
    # Calculate properties for first few valid molecules
    print("Property examples:")
    for i, smiles in enumerate(processed_smiles[:3]):
        props = property_calc.calculate_all_properties(smiles)
        print(f"\n  {smiles}:")
        print(f"    Molecular Weight: {props.get('molecular_weight', 'N/A'):.2f}")
        print(f"    LogP: {props.get('logp', 'N/A'):.2f}")
        print(f"    TPSA: {props.get('tpsa', 'N/A'):.2f}")
        print(f"    QED: {props.get('qed', 'N/A'):.3f}")
        print(f"    Drug-like: {props.get('drug_like', 'N/A')}")
    
    # 4. Dataset Building
    print("\n4. Complete Dataset Building")
    print("-" * 28)
    
    # Create data directory
    data_dir = Path("./example_data")
    data_dir.mkdir(exist_ok=True)
    
    # Save sample data to file
    raw_data_file = data_dir / "raw_molecules.txt"
    with open(raw_data_file, "w") as f:
        for smiles in sample_smiles:
            f.write(f"{smiles}\n")
    
    print(f"Saved raw data to: {raw_data_file}")
    
    # Build complete dataset
    dataset_builder = DatasetBuilder(str(data_dir))
    
    try:
        dataset_info = dataset_builder.build_complete_dataset(
            raw_smiles_path=str(raw_data_file),
            dataset_name="example_dataset",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            create_reasoning_data=True,
            n_reasoning_examples=50  # Small number for example
        )
        
        print("\nDataset building completed!")
        print(f"Total molecules processed: {dataset_info['total_molecules']}")
        print(f"Training molecules: {dataset_info['processing_stats']['final_count']}")
        
        # Show split information
        if 'split_paths' in dataset_info:
            for split_name, split_path in dataset_info['split_paths'].items():
                print(f"{split_name.capitalize()} set: {split_path}")
        
        # Show reasoning data statistics
        if 'reasoning_stats' in dataset_info:
            reasoning_stats = dataset_info['reasoning_stats']
            print(f"\nReasoning data created:")
            for task_type, stats in reasoning_stats.items():
                if isinstance(stats, dict) and 'total_examples' in stats:
                    print(f"  {task_type}: {stats['total_examples']} examples")
    
    except Exception as e:
        print(f"Dataset building failed: {e}")
        print("This is expected in the example due to simplified setup")
    
    # 5. Data Loader Example
    print("\n5. Data Loader Setup")
    print("-" * 20)
    
    from molgeneration.data.loader import DataLoader
    from molgeneration.utils.smiles_utils import SMILESTokenizer
    
    # Initialize tokenizer and data loader
    tokenizer = SMILESTokenizer()
    data_loader = DataLoader(tokenizer)
    
    print(f"Tokenizer vocabulary size: {data_loader.get_vocab_size()}")
    
    # Show tokenization example
    example_smiles = "CCO"
    tokens = tokenizer.encode(example_smiles)
    decoded = tokenizer.decode(tokens)
    
    print(f"\nTokenization example:")
    print(f"  Original: {example_smiles}")
    print(f"  Tokens: {tokens}")
    print(f"  Decoded: {decoded}")
    
    # 6. Chemical Analysis
    print("\n6. Chemical Analysis Examples")
    print("-" * 30)
    
    from molgeneration.utils.chemical_utils import MolecularSimilarity, MolecularFragments
    
    # Similarity analysis
    similarity_calc = MolecularSimilarity()
    
    mol1, mol2 = "CCO", "CC(C)O"
    similarity = similarity_calc.calculate_similarity(mol1, mol2)
    print(f"Similarity between {mol1} and {mol2}: {similarity:.3f}")
    
    # Fragment analysis
    fragment_analyzer = MolecularFragments()
    
    example_mol = "COc1ccccc1"
    fragments = fragment_analyzer.count_fragments(example_mol)
    print(f"\nFragment analysis for {example_mol}:")
    for fragment_type, count in fragments.items():
        if count and count > 0:
            print(f"  {fragment_type}: {count}")
    
    print("\nData processing example completed!")
    print(f"Check the '{data_dir}' directory for generated files.")


if __name__ == "__main__":
    main()