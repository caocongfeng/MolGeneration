"""Example script demonstrating how to train a molecular generation model."""

import torch
from pathlib import Path
from molgeneration.config.config import MolGenerationConfig
from molgeneration.data.loader import DataLoader
from molgeneration.data.preprocessing import DatasetBuilder
from molgeneration.training.two_stage_trainer import TwoStageTrainer
from molgeneration.evaluation.evaluator import MolecularEvaluator
from molgeneration.utils.smiles_utils import SMILESTokenizer


def main():
    """Main training pipeline example."""
    
    # 1. Setup configuration
    print("Setting up configuration...")
    config = MolGenerationConfig()
    
    # Customize configuration
    config.experiment_name = "mol_generation_example"
    config.output_dir = "./outputs"
    config.training.stage1_epochs = 3
    config.training.stage2_epochs = 3
    config.training.batch_size = 16
    config.model.num_layers = 6
    config.model.hidden_size = 512
    
    # Save configuration
    config.to_yaml("./config.yaml")
    print(f"Configuration saved to config.yaml")
    
    # 2. Prepare dataset (example with dummy data)
    print("\nPreparing dataset...")
    
    # Create dummy SMILES data
    dummy_smiles = [
        "CCO",
        "CC(C)O",
        "C1CCCCC1",
        "c1ccccc1",
        "CCC(=O)O",
        "CCN(CC)CC",
        "COc1ccccc1",
        "CC(C)(C)O",
        "CCCCO",
        "CC(O)CC"
    ] * 100  # Duplicate for more data
    
    # Save dummy data
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    with open(data_dir / "raw_smiles.txt", "w") as f:
        for smiles in dummy_smiles:
            f.write(f"{smiles}\n")
    
    # Build dataset
    dataset_builder = DatasetBuilder("./data")
    dataset_info = dataset_builder.build_complete_dataset(
        raw_smiles_path="./data/raw_smiles.txt",
        dataset_name="example_dataset",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        create_reasoning_data=True,
        n_reasoning_examples=100
    )
    
    print(f"Dataset built: {dataset_info['total_molecules']} molecules")
    
    # 3. Setup tokenizer and data loaders
    print("\nSetting up tokenizer and data loaders...")
    tokenizer = SMILESTokenizer()
    data_loader = DataLoader(tokenizer)
    
    vocab_size = data_loader.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    # 4. Initialize trainer
    print("\nInitializing two-stage trainer...")
    trainer = TwoStageTrainer(
        config=config,
        vocab_size=vocab_size,
        device="cpu"  # Use CPU for this example
    )
    
    # 5. Setup data loaders for training
    reasoning_data_loader = data_loader
    generation_data_loader = data_loader
    reference_molecules = dummy_smiles[:100]  # Use subset as reference
    
    # 6. Train the model
    print("\nStarting two-stage training...")
    try:
        training_results = trainer.train(
            reasoning_data_loader=reasoning_data_loader,
            generation_data_loader=generation_data_loader,
            reference_molecules=reference_molecules,
            resume_from_checkpoint=None,
            skip_stage1=False,
            skip_stage2=False
        )
        
        print("\nTraining completed successfully!")
        print(f"Stage 1 completed: {training_results.get('stage1_results') is not None}")
        print(f"Stage 2 completed: {training_results.get('stage2_results') is not None}")
        
        # 7. Save the final model
        print("\nSaving final model...")
        trainer.save_final_model("./outputs/final_model")
        
        # 8. Final evaluation
        if training_results.get('final_evaluation'):
            evaluation = training_results['final_evaluation']
            print("\nFinal Evaluation Results:")
            for metric, value in evaluation['metrics'].items():
                print(f"  {metric}: {value:.4f}")
        
        # Print training summary
        print("\n" + trainer.get_training_summary())
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


def inference_example():
    """Example of using a trained model for inference."""
    
    print("\n" + "="*50)
    print("INFERENCE EXAMPLE")
    print("="*50)
    
    # Load configuration
    config = MolGenerationConfig.from_yaml("./config.yaml")
    
    # Setup tokenizer
    tokenizer = SMILESTokenizer()
    vocab_size = len(tokenizer.vocab)
    
    # Load trained model (this would load from saved checkpoint)
    from molgeneration.models.mol_generation_model import MolGenerationModel
    
    model = MolGenerationModel(config.model, vocab_size)
    model.eval()
    
    # Generate molecules
    print("Generating molecules...")
    
    with torch.no_grad():
        # Start tokens
        input_ids = torch.tensor([[2]], dtype=torch.long)  # Start token
        
        # Generate
        generated = model.generate(
            input_ids=input_ids,
            max_length=config.model.generation_max_length,
            temperature=config.model.generation_temperature,
            top_k=config.model.generation_top_k,
            top_p=config.model.generation_top_p,
            do_sample=True,
            num_return_sequences=5
        )
        
        print(f"Generated {len(generated)} molecules")
        
        # Evaluate generated molecules
        evaluator = MolecularEvaluator()
        
        # Decode sequences (simplified)
        generated_smiles = [f"Generated_Molecule_{i}" for i in range(len(generated))]
        
        evaluation_results = evaluator.evaluate(
            generated_molecules=generated_smiles,
            reference_molecules=["CCO", "CC(C)O", "C1CCCCC1"],
            detailed_analysis=True
        )
        
        print("\nGeneration Evaluation:")
        print(evaluator.get_metric_summary(evaluation_results))


def evaluation_example():
    """Example of evaluating generated molecules."""
    
    print("\n" + "="*50)
    print("EVALUATION EXAMPLE")
    print("="*50)
    
    # Sample generated molecules
    generated_molecules = [
        "CCO",
        "CC(C)O", 
        "CCC",
        "C1CCCCC1",
        "c1ccccc1",
        "Invalid_SMILES",  # This should fail validation
        "CCC(=O)O",
        "CCN(CC)CC"
    ]
    
    # Reference molecules
    reference_molecules = [
        "CCO",
        "CC(O)CC",
        "CCCC",
        "C1CCCC1"
    ]
    
    # Initialize evaluator
    evaluator = MolecularEvaluator(
        reference_molecules=reference_molecules,
        target_properties={'qed': 0.5, 'molecular_weight': 100.0},
        include_property_metrics=True,
        include_diversity_metrics=True
    )
    
    # Evaluate
    results = evaluator.evaluate(
        generated_molecules=generated_molecules,
        detailed_analysis=True,
        save_path="./evaluation_results.json"
    )
    
    print("Evaluation completed!")
    print(evaluator.get_metric_summary(results))
    
    # Compare multiple models (example)
    model_results = {
        "Model_A": generated_molecules[:4],
        "Model_B": generated_molecules[4:],
        "Baseline": ["CC", "CCC", "CCCC"]
    }
    
    comparison = evaluator.compare_models(
        model_results=model_results,
        reference_molecules=reference_molecules,
        save_path="./model_comparison.json"
    )
    
    print(f"\nModel comparison completed. Best model: {comparison['best_model']}")


if __name__ == "__main__":
    print("MolGeneration Example")
    print("=" * 50)
    
    # Run main training example
    main()
    
    # Run inference example
    inference_example()
    
    # Run evaluation example
    evaluation_example()
    
    print("\nAll examples completed!")