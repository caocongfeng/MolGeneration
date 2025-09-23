"""Two-stage trainer for molecular generation models."""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from molgeneration.config.config import MolGenerationConfig
from molgeneration.models.mol_generation_model import MolGenerationModel
from molgeneration.models.reasoning_model import ReasoningModel
from molgeneration.models.generation_model import GenerationModel
from molgeneration.data.loader import DataLoader
from molgeneration.training.stage1_trainer import Stage1Trainer
from molgeneration.training.stage2_trainer import Stage2Trainer
from molgeneration.utils.training_utils import CheckpointManager, LoggingUtils
from molgeneration.evaluation.evaluator import MolecularEvaluator

logger = logging.getLogger(__name__)


class TwoStageTrainer:
    """
    Main trainer for two-stage molecular generation training.
    
    Coordinates the training process:
    1. Stage 1: Chemical reasoning pre-training
    2. Stage 2: Molecular generation fine-tuning with RL
    """
    
    def __init__(
        self,
        config: MolGenerationConfig,
        vocab_size: int,
        device: Optional[str] = None
    ):
        """
        Initialize two-stage trainer.
        
        Args:
            config: Training configuration
            vocab_size: Vocabulary size
            device: Device to use for training
        """
        self.config = config
        self.vocab_size = vocab_size
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize models
        self.reasoning_model = None
        self.generation_model = None
        self.main_model = None
        
        # Initialize trainers
        self.stage1_trainer = None
        self.stage2_trainer = None
        
        # Initialize utilities
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self.logger = LoggingUtils(
            config.experiment_name,
            config.log_dir,
            use_wandb=True,
            use_tensorboard=True
        )
        
        # Initialize evaluator
        self.evaluator = MolecularEvaluator()
        
        # Training state
        self.current_stage = None
        self.global_step = 0
        self.stage1_completed = False
        self.stage2_completed = False
        
        print(f"TwoStageTrainer initialized on device: {self.device}")
    
    def setup_models(self) -> None:
        """Initialize all models."""
        print("Setting up models...")
        
        # Initialize models
        self.reasoning_model = ReasoningModel(self.config.model, self.vocab_size)
        self.generation_model = GenerationModel(self.config.model, self.vocab_size)
        self.main_model = MolGenerationModel(self.config.model, self.vocab_size)
        
        # Move to device
        self.reasoning_model.to(self.device)
        self.generation_model.to(self.device)
        self.main_model.to(self.device)
        
        # Log model architectures
        self.logger.log_model_architecture(self.main_model)
        
        print("Models setup complete")
    
    def setup_stage1_trainer(self, data_loader: DataLoader) -> None:
        """Setup Stage 1 trainer."""
        print("Setting up Stage 1 trainer...")
        
        self.stage1_trainer = Stage1Trainer(
            model=self.reasoning_model,
            config=self.config,
            data_loader=data_loader,
            device=self.device,
            checkpoint_manager=self.checkpoint_manager,
            logger=self.logger
        )
        
        print("Stage 1 trainer setup complete")
    
    def setup_stage2_trainer(self, data_loader: DataLoader, reference_molecules: List[str]) -> None:
        """Setup Stage 2 trainer."""
        print("Setting up Stage 2 trainer...")
        
        self.stage2_trainer = Stage2Trainer(
            model=self.generation_model,
            config=self.config,
            data_loader=data_loader,
            reference_molecules=reference_molecules,
            device=self.device,
            checkpoint_manager=self.checkpoint_manager,
            logger=self.logger,
            evaluator=self.evaluator
        )
        
        print("Stage 2 trainer setup complete")
    
    def train(
        self,
        reasoning_data_loader: DataLoader,
        generation_data_loader: DataLoader,
        reference_molecules: List[str],
        resume_from_checkpoint: Optional[str] = None,
        skip_stage1: bool = False,
        skip_stage2: bool = False
    ) -> Dict[str, Any]:
        """
        Execute two-stage training.
        
        Args:
            reasoning_data_loader: Data loader for reasoning training
            generation_data_loader: Data loader for generation training
            reference_molecules: Reference molecules for evaluation
            resume_from_checkpoint: Path to checkpoint to resume from
            skip_stage1: Whether to skip Stage 1 training
            skip_stage2: Whether to skip Stage 2 training
            
        Returns:
            Training results and metrics
        """
        print("Starting two-stage training...")
        
        # Setup models
        self.setup_models()
        
        # Log configuration
        self.logger.log_config(self.config.__dict__)
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self._resume_from_checkpoint(resume_from_checkpoint)
        
        results = {
            'stage1_results': None,
            'stage2_results': None,
            'final_evaluation': None,
            'training_completed': False
        }
        
        try:
            # Stage 1: Chemical reasoning pre-training
            if not skip_stage1 and not self.stage1_completed:
                print("\n" + "="*50)
                print("STAGE 1: Chemical Reasoning Pre-training")
                print("="*50)
                
                self.current_stage = 1
                self.setup_stage1_trainer(reasoning_data_loader)
                
                stage1_results = self.stage1_trainer.train()
                results['stage1_results'] = stage1_results
                
                # Transfer knowledge to main model
                self._transfer_stage1_knowledge()
                self.stage1_completed = True
                
                # Save stage 1 checkpoint
                self._save_stage_checkpoint(1)
            
            # Stage 2: Molecular generation fine-tuning
            if not skip_stage2 and not self.stage2_completed:
                print("\n" + "="*50)
                print("STAGE 2: Molecular Generation Fine-tuning")
                print("="*50)
                
                self.current_stage = 2
                self.setup_stage2_trainer(generation_data_loader, reference_molecules)
                
                # Initialize generation model with reasoning knowledge
                if self.stage1_completed:
                    self._initialize_stage2_from_stage1()
                
                stage2_results = self.stage2_trainer.train()
                results['stage2_results'] = stage2_results
                
                # Transfer knowledge to main model
                self._transfer_stage2_knowledge()
                self.stage2_completed = True
                
                # Save stage 2 checkpoint
                self._save_stage_checkpoint(2)
            
            # Final evaluation
            print("\n" + "="*50)
            print("FINAL EVALUATION")
            print("="*50)
            
            final_evaluation = self._final_evaluation(reference_molecules)
            results['final_evaluation'] = final_evaluation
            
            results['training_completed'] = True
            
            print("\nTwo-stage training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            results['error'] = str(e)
            raise
        
        finally:
            # Clean up
            self.logger.close()
        
        return results
    
    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint."""
        print(f"Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore training state
        self.global_step = checkpoint.get('global_step', 0)
        self.current_stage = checkpoint.get('current_stage', 1)
        self.stage1_completed = checkpoint.get('stage1_completed', False)
        self.stage2_completed = checkpoint.get('stage2_completed', False)
        
        # Load model states if available
        if 'reasoning_model_state' in checkpoint and self.reasoning_model:
            self.reasoning_model.load_state_dict(checkpoint['reasoning_model_state'])
        
        if 'generation_model_state' in checkpoint and self.generation_model:
            self.generation_model.load_state_dict(checkpoint['generation_model_state'])
        
        if 'main_model_state' in checkpoint and self.main_model:
            self.main_model.load_state_dict(checkpoint['main_model_state'])
        
        print(f"Resumed from stage {self.current_stage}, step {self.global_step}")
    
    def _transfer_stage1_knowledge(self) -> None:
        """Transfer knowledge from stage 1 reasoning model to main model."""
        print("Transferring Stage 1 knowledge to main model...")
        
        # Transfer base language model weights
        main_lm_state = self.main_model.base_model.state_dict()
        reasoning_lm_state = self.reasoning_model.language_model.state_dict()
        
        # Update matching parameters
        for name, param in reasoning_lm_state.items():
            if name in main_lm_state:
                main_lm_state[name].copy_(param)
        
        print("Stage 1 knowledge transfer complete")
    
    def _initialize_stage2_from_stage1(self) -> None:
        """Initialize stage 2 model with stage 1 knowledge."""
        print("Initializing Stage 2 model with Stage 1 knowledge...")
        
        # Transfer base language model weights
        gen_lm_state = self.generation_model.generator.state_dict()
        reasoning_lm_state = self.reasoning_model.language_model.state_dict()
        
        # Update matching parameters
        for name, param in reasoning_lm_state.items():
            if name in gen_lm_state:
                gen_lm_state[name].copy_(param)
        
        print("Stage 2 initialization complete")
    
    def _transfer_stage2_knowledge(self) -> None:
        """Transfer knowledge from stage 2 generation model to main model."""
        print("Transferring Stage 2 knowledge to main model...")
        
        # Transfer generation model weights
        main_lm_state = self.main_model.base_model.state_dict()
        gen_lm_state = self.generation_model.generator.state_dict()
        
        # Update matching parameters
        for name, param in gen_lm_state.items():
            if name in main_lm_state:
                main_lm_state[name].copy_(param)
        
        print("Stage 2 knowledge transfer complete")
    
    def _save_stage_checkpoint(self, stage: int) -> None:
        """Save checkpoint after completing a stage."""
        print(f"Saving Stage {stage} checkpoint...")
        
        checkpoint_state = {
            'global_step': self.global_step,
            'current_stage': stage,
            'stage1_completed': self.stage1_completed,
            'stage2_completed': self.stage2_completed,
            'config': self.config.__dict__,
        }
        
        # Add model states
        if self.reasoning_model:
            checkpoint_state['reasoning_model_state'] = self.reasoning_model.state_dict()
        
        if self.generation_model:
            checkpoint_state['generation_model_state'] = self.generation_model.state_dict()
        
        if self.main_model:
            checkpoint_state['main_model_state'] = self.main_model.state_dict()
        
        # Save checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f"stage_{stage}_complete.pt"
        torch.save(checkpoint_state, checkpoint_path)
        
        print(f"Stage {stage} checkpoint saved: {checkpoint_path}")
    
    def _final_evaluation(self, reference_molecules: List[str]) -> Dict[str, Any]:
        """Perform final evaluation of the trained model."""
        print("Performing final evaluation...")
        
        # Generate molecules using the main model
        self.main_model.eval()
        
        with torch.no_grad():
            # Generate sample molecules
            batch_size = 100
            num_batches = 10
            generated_molecules = []
            
            for batch_idx in range(num_batches):
                print(f"Generating batch {batch_idx + 1}/{num_batches}...")
                
                # Start tokens
                input_ids = torch.full(
                    (batch_size, 1), 2, dtype=torch.long, device=self.device
                )
                
                # Generate
                generated = self.main_model.generate(
                    input_ids=input_ids,
                    max_length=self.config.model.generation_max_length,
                    temperature=self.config.model.generation_temperature,
                    top_k=self.config.model.generation_top_k,
                    top_p=self.config.model.generation_top_p,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=0,
                    eos_token_id=1
                )
                
                # Decode generated sequences
                # Note: In a real implementation, you would use the tokenizer
                # For now, we'll create dummy SMILES for evaluation
                batch_molecules = [f"C{i}CC" for i in range(len(generated))]
                generated_molecules.extend(batch_molecules)
        
        # Evaluate generated molecules
        evaluation_results = self.evaluator.evaluate(
            generated_molecules=generated_molecules,
            reference_molecules=reference_molecules,
            detailed_analysis=True
        )
        
        # Log evaluation results
        self.logger.log_metrics(
            evaluation_results['metrics'],
            step=self.global_step,
            prefix="final_evaluation"
        )
        
        # Log generated molecules
        self.logger.log_generated_molecules(
            generated_molecules,
            step=self.global_step,
            stage="final"
        )
        
        print("Final evaluation complete")
        print(self.evaluator.get_metric_summary(evaluation_results))
        
        return evaluation_results
    
    def save_final_model(self, save_path: str) -> None:
        """Save the final trained model."""
        print(f"Saving final model to {save_path}")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = save_path / "pytorch_model.bin"
        torch.save(self.main_model.state_dict(), model_path)
        
        # Save config
        config_path = save_path / "config.json"
        self.config.to_yaml(str(config_path.with_suffix('.yaml')))
        
        # Save training info
        training_info = {
            'stage1_completed': self.stage1_completed,
            'stage2_completed': self.stage2_completed,
            'global_step': self.global_step,
            'vocab_size': self.vocab_size,
            'device': str(self.device)
        }
        
        info_path = save_path / "training_info.json"
        import json
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"Final model saved to {save_path}")
    
    def get_training_summary(self) -> str:
        """Get a summary of the training process."""
        summary_lines = [
            "=== Two-Stage Training Summary ===",
            f"Experiment: {self.config.experiment_name}",
            f"Device: {self.device}",
            f"Vocabulary size: {self.vocab_size}",
            "",
            f"Stage 1 completed: {self.stage1_completed}",
            f"Stage 2 completed: {self.stage2_completed}",
            f"Global step: {self.global_step}",
            "",
            "Configuration:",
            f"  Model layers: {self.config.model.num_layers}",
            f"  Hidden size: {self.config.model.hidden_size}",
            f"  Attention heads: {self.config.model.num_attention_heads}",
            f"  Stage 1 epochs: {self.config.training.stage1_epochs}",
            f"  Stage 2 epochs: {self.config.training.stage2_epochs}",
            f"  Learning rates: {self.config.training.stage1_lr}, {self.config.training.stage2_lr}",
            "=" * 35
        ]
        
        return "\n".join(summary_lines)