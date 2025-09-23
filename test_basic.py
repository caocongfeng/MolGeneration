"""Basic test script to validate MolGeneration functionality."""

import sys
import traceback
from pathlib import Path


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Core modules
        from molgeneration.config.config import MolGenerationConfig
        from molgeneration.data.loader import DataLoader
        from molgeneration.models.mol_generation_model import MolGenerationModel
        from molgeneration.training.two_stage_trainer import TwoStageTrainer
        from molgeneration.evaluation.evaluator import MolecularEvaluator
        
        # Utilities
        from molgeneration.utils.smiles_utils import SMILESValidator, SMILESTokenizer
        from molgeneration.utils.chemical_utils import PropertyCalculator
        from molgeneration.utils.training_utils import CheckpointManager
        
        print("✓ All imports successful")
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration...")
    
    try:
        config = MolGenerationConfig()
        
        # Test basic properties
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'data')
        
        # Test configuration updates
        config.experiment_name = "test_experiment"
        assert config.experiment_name == "test_experiment"
        
        # Test YAML export/import
        test_yaml = "/tmp/test_config.yaml"
        config.to_yaml(test_yaml)
        
        loaded_config = MolGenerationConfig.from_yaml(test_yaml)
        assert loaded_config.experiment_name == "test_experiment"
        
        print("✓ Configuration system working")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_smiles_processing():
    """Test SMILES processing utilities."""
    print("\nTesting SMILES processing...")
    
    try:
        from molgeneration.utils.smiles_utils import SMILESValidator, SMILESCanonicalizer, SMILESTokenizer
        
        # Test validation
        validator = SMILESValidator()
        assert validator.is_valid("CCO") == True
        assert validator.is_valid("INVALID") == False
        
        # Test canonicalization
        canonicalizer = SMILESCanonicalizer()
        canonical = canonicalizer.canonicalize("CCO")
        assert canonical is not None
        
        # Test tokenization
        tokenizer = SMILESTokenizer()
        tokens = tokenizer.encode("CCO")
        decoded = tokenizer.decode(tokens)
        assert isinstance(tokens, list)
        assert isinstance(decoded, str)
        
        print("✓ SMILES processing working")
        return True
        
    except Exception as e:
        print(f"✗ SMILES processing test failed: {e}")
        traceback.print_exc()
        return False


def test_chemical_utils():
    """Test chemical utility functions.""" 
    print("\nTesting chemical utilities...")
    
    try:
        from molgeneration.utils.chemical_utils import PropertyCalculator, MolecularSimilarity
        
        # Test property calculation
        prop_calc = PropertyCalculator()
        props = prop_calc.calculate_all_properties("CCO")
        
        assert isinstance(props, dict)
        assert 'molecular_weight' in props
        
        # Test similarity calculation
        sim_calc = MolecularSimilarity()
        similarity = sim_calc.calculate_similarity("CCO", "CC(C)O")
        
        if similarity is not None:
            assert 0 <= similarity <= 1
        
        print("✓ Chemical utilities working")
        return True
        
    except Exception as e:
        print(f"✗ Chemical utilities test failed: {e}")
        traceback.print_exc()
        return False


def test_model_initialization():
    """Test model initialization."""
    print("\nTesting model initialization...")
    
    try:
        from molgeneration.config.config import MolGenerationConfig, ModelConfig
        from molgeneration.models.mol_generation_model import MolGenerationModel
        from molgeneration.models.reasoning_model import ReasoningModel
        from molgeneration.models.generation_model import GenerationModel
        
        # Create small config for testing
        config = ModelConfig()
        config.hidden_size = 64
        config.num_layers = 2
        config.num_attention_heads = 2
        
        vocab_size = 100
        
        # Test main model
        main_model = MolGenerationModel(config, vocab_size)
        assert main_model is not None
        
        # Test reasoning model
        reasoning_model = ReasoningModel(config, vocab_size)
        assert reasoning_model is not None
        
        # Test generation model
        generation_model = GenerationModel(config, vocab_size)
        assert generation_model is not None
        
        print("✓ Model initialization working")
        return True
        
    except Exception as e:
        print(f"✗ Model initialization test failed: {e}")
        traceback.print_exc()
        return False


def test_evaluation():
    """Test evaluation system."""
    print("\nTesting evaluation system...")
    
    try:
        from molgeneration.evaluation.evaluator import MolecularEvaluator
        from molgeneration.evaluation.metrics import ValidityMetric, UniquenessMetric
        
        # Test individual metrics
        validity_metric = ValidityMetric()
        validity = validity_metric.compute(["CCO", "CC(C)O", "INVALID"])
        assert 0 <= validity <= 1
        
        uniqueness_metric = UniquenessMetric()
        uniqueness = uniqueness_metric.compute(["CCO", "CCO", "CC(C)O"])
        assert 0 <= uniqueness <= 1
        
        # Test evaluator
        evaluator = MolecularEvaluator()
        results = evaluator.evaluate(
            generated_molecules=["CCO", "CC(C)O"],
            reference_molecules=["CCC", "CCCC"],
            detailed_analysis=False
        )
        
        assert 'metrics' in results
        assert 'validity' in results['metrics']
        
        print("✓ Evaluation system working")
        return True
        
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        traceback.print_exc()
        return False


def test_data_processing():
    """Test data processing components."""
    print("\nTesting data processing...")
    
    try:
        from molgeneration.data.preprocessing import SMILESProcessor
        from molgeneration.data.loader import DataLoader
        from molgeneration.utils.smiles_utils import SMILESTokenizer
        
        # Test SMILES processor
        processor = SMILESProcessor()
        processed, indices = processor.process_smiles_list(["CCO", "INVALID", "CC(C)O"])
        assert len(processed) >= 2  # Should filter out invalid
        
        # Test data loader
        tokenizer = SMILESTokenizer()
        data_loader = DataLoader(tokenizer)
        vocab_size = data_loader.get_vocab_size()
        assert vocab_size > 0
        
        print("✓ Data processing working")
        return True
        
    except Exception as e:
        print(f"✗ Data processing test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("MolGeneration Basic Functionality Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_configuration,
        test_smiles_processing,
        test_chemical_utils,
        test_model_initialization,
        test_evaluation,
        test_data_processing,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n{'='*40}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MolGeneration is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())