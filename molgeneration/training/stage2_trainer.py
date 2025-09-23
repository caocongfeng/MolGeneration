"""Stage 2 trainer for molecular generation fine-tuning with reinforcement learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from tqdm import tqdm
import logging
from collections import deque

from molgeneration.config.config import MolGenerationConfig
from molgeneration.models.generation_model import GenerationModel
from molgeneration.data.loader import DataLoader
from molgeneration.utils.training_utils import CheckpointManager, LoggingUtils
from molgeneration.evaluation.evaluator import MolecularEvaluator
from molgeneration.utils.smiles_utils import SMILESValidator
from molgeneration.utils.chemical_utils import PropertyCalculator

logger = logging.getLogger(__name__)


class Stage2Trainer:
    """
    Trainer for Stage 2: Molecular generation fine-tuning with reinforcement learning.
    
    Trains the model for controlled molecule generation using property-based
    rewards and reinforcement learning techniques.
    """
    
    def __init__(
        self,
        model: GenerationModel,
        config: MolGenerationConfig,
        data_loader: DataLoader,
        reference_molecules: List[str],
        device: torch.device,
        checkpoint_manager: CheckpointManager,
        logger: LoggingUtils,
        evaluator: MolecularEvaluator
    ):
        """
        Initialize Stage 2 trainer.
        
        Args:
            model: Generation model to train
            config: Training configuration
            data_loader: Data loader for generation training
            reference_molecules: Reference molecules for evaluation
            device: Training device
            checkpoint_manager: Checkpoint manager
            logger: Logging utilities
            evaluator: Molecular evaluator
        """
        self.model = model
        self.config = config
        self.data_loader = data_loader
        self.reference_molecules = reference_molecules
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger
        self.evaluator = evaluator
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_reward = -float('inf')
        
        # Setup optimizers and schedulers
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # RL components
        self.reward_calculator = RewardCalculator()
        self.experience_buffer = ExperienceBuffer(max_size=10000)
        
        # Training history
        self.reward_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        
        # Evaluation frequency
        self.eval_frequency = 500  # Steps
        
        print("Stage 2 trainer initialized")
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer for Stage 2 training."""
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.stage2_lr,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        return optimizer
    
    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        total_steps = self.config.training.stage2_epochs * 1000  # Approximate
        warmup_steps = min(self.config.training.warmup_steps, total_steps // 10)
        
        scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        return scheduler
    
    def train(self) -> Dict[str, Any]:
        """
        Execute Stage 2 training.
        
        Returns:
            Training results and metrics
        """
        print(f"Starting Stage 2 training for {self.config.training.stage2_epochs} epochs...")
        
        training_results = {
            'epochs_completed': 0,
            'best_reward': -float('inf'),
            'final_metrics': {},
            'training_history': [],
            'generated_molecules': []
        }
        
        for epoch in range(self.config.training.stage2_epochs):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch + 1}/{self.config.training.stage2_epochs}")
            
            # Training step
            epoch_metrics = self._train_epoch()
            
            # Evaluation step
            eval_metrics = self._evaluate_generation()
            
            # Combine metrics
            combined_metrics = {
                **{f"train_{k}": v for k, v in epoch_metrics.items()},
                **{f"eval_{k}": v for k, v in eval_metrics.items()}
            }
            
            # Log metrics
            self.logger.log_metrics(
                combined_metrics,
                step=self.global_step,
                epoch=epoch,
                prefix="stage2"
            )
            
            # Save checkpoint if best model
            current_reward = eval_metrics.get('average_reward', -float('inf'))
            is_best = current_reward > self.best_reward
            if is_best:
                self.best_reward = current_reward
                self._save_checkpoint(is_best=True)
            
            # Regular checkpoint
            if (epoch + 1) % self.config.training.save_steps == 0:
                self._save_checkpoint(is_best=False)
            
            # Update training history
            training_results['training_history'].append({
                'epoch': epoch,
                'train_loss': epoch_metrics.get('total_loss', 0.0),
                'average_reward': current_reward,
                'validity': eval_metrics.get('validity', 0.0),
                'uniqueness': eval_metrics.get('uniqueness', 0.0),
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            training_results['epochs_completed'] = epoch + 1
        
        training_results['best_reward'] = self.best_reward
        training_results['final_metrics'] = combined_metrics
        
        # Final molecule generation
        final_molecules = self._generate_final_molecules()
        training_results['generated_molecules'] = final_molecules
        
        print(f"Stage 2 training completed. Best reward: {self.best_reward:.4f}")
        
        return training_results
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch using reinforcement learning."""
        self.model.train()
        
        epoch_losses = {
            'total_loss': [],
            'generation_loss': [],
            'rl_loss': [],
            'value_loss': [],
            'policy_loss': [],
            'entropy_loss': []
        }
        
        epoch_rewards = []
        
        # Create generation data loader
        generation_dataloader = self.data_loader.create_generation_dataloader(
            data_path="data/generation/train.csv",  # Placeholder
            batch_size=self.config.training.batch_size,
            shuffle=True,
            split="train"
        )
        
        progress_bar = tqdm(generation_dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            if self.config.training.use_rl and batch_idx % 2 == 0:
                # Reinforcement learning step
                rl_metrics = self._rl_training_step(batch)
                
                for loss_name, loss_value in rl_metrics.items():
                    if loss_name in epoch_losses:
                        epoch_losses[loss_name].append(loss_value)
                
                if 'average_reward' in rl_metrics:
                    epoch_rewards.append(rl_metrics['average_reward'])
            
            else:
                # Supervised learning step
                sl_metrics = self._supervised_training_step(batch)
                
                for loss_name, loss_value in sl_metrics.items():
                    if loss_name in epoch_losses:
                        epoch_losses[loss_name].append(loss_value)
            
            # Update progress bar
            current_loss = epoch_losses['total_loss'][-1] if epoch_losses['total_loss'] else 0.0
            current_reward = epoch_rewards[-1] if epoch_rewards else 0.0
            
            progress_bar.set_postfix({
                'loss': f"{current_loss:.4f}",
                'reward': f"{current_reward:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Periodic evaluation
            if self.global_step % self.eval_frequency == 0:
                self._periodic_evaluation()
        
        # Average metrics for epoch
        epoch_metrics = {}
        for loss_name, loss_values in epoch_losses.items():
            if loss_values:
                epoch_metrics[loss_name] = np.mean(loss_values)
        
        if epoch_rewards:
            epoch_metrics['average_reward'] = np.mean(epoch_rewards)
            self.reward_history.extend(epoch_rewards)
        
        return epoch_metrics
    
    def _supervised_training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform supervised learning training step."""
        # Extract target properties if available
        target_properties = {}
        for prop_name in ['molecular_weight', 'logp', 'tpsa', 'qed']:
            if f'target_{prop_name}' in batch:
                target_properties[prop_name] = batch[f'target_{prop_name}']
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            labels=batch.get('labels'),
            target_properties=target_properties if target_properties else None,
            return_generation_outputs=True
        )
        
        # Compute loss
        total_loss = outputs.get('generation_loss', torch.tensor(0.0, device=self.device))
        
        # Add property conditioning loss if available
        if target_properties and 'property_outputs' in outputs:
            property_loss = torch.tensor(0.0, device=self.device)
            for prop_name, target_value in target_properties.items():
                if prop_name in outputs['property_outputs']:
                    prop_pred = outputs['property_outputs'][prop_name]
                    # Simplified property loss
                    property_loss += F.mse_loss(prop_pred.mean(dim=1), target_value)
            
            total_loss += 0.1 * property_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
        self.optimizer.step()
        self.global_step += 1
        
        return {
            'total_loss': total_loss.item(),
            'generation_loss': outputs.get('generation_loss', torch.tensor(0.0)).item()
        }
    
    def _rl_training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform reinforcement learning training step."""
        batch_size = batch['input_ids'].shape[0]
        
        # Generate molecules
        with torch.no_grad():
            generated_data = self.model.generate_molecules(
                batch_size=batch_size,
                max_length=self.config.model.generation_max_length,
                temperature=self.config.model.generation_temperature,
                device=self.device
            )
        
        sequences = generated_data['sequences']
        metadata = generated_data['metadata']
        
        # Compute rewards
        rewards = self._compute_rewards(sequences)
        
        # Convert sequences to tensor format for loss computation
        if isinstance(sequences, list):
            # Convert list of tensors to single tensor
            max_len = max(seq.shape[1] for seq in sequences)
            padded_sequences = torch.zeros(len(sequences), max_len, dtype=torch.long, device=self.device)
            for i, seq in enumerate(sequences):
                padded_sequences[i, :seq.shape[1]] = seq[0]  # Take first sequence
            sequences = padded_sequences
        
        # Get old policy log probs (simplified - should be computed during generation)
        old_log_probs = torch.randn_like(sequences.float())  # Placeholder
        
        # Compute value estimates
        outputs = self.model(
            input_ids=sequences,
            return_generation_outputs=True,
            compute_rewards=True
        )
        
        values = outputs.get('values', torch.zeros(batch_size, device=self.device))
        
        # Compute RL loss
        rl_losses = self.model.compute_rl_loss(
            sequences=sequences,
            rewards=rewards,
            values=values,
            old_log_probs=old_log_probs,
            clip_range=self.config.training.rl_clip_range
        )
        
        total_loss = rl_losses['total_rl_loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
        self.optimizer.step()
        self.global_step += 1
        
        # Store experience
        self.experience_buffer.add_experience(sequences, rewards, values)
        
        metrics = {
            'total_loss': total_loss.item(),
            'rl_loss': rl_losses['total_rl_loss'].item(),
            'policy_loss': rl_losses['policy_loss'].item(),
            'value_loss': rl_losses['value_loss'].item(),
            'entropy_loss': rl_losses['entropy'].item(),
            'average_reward': rewards.mean().item(),
            'reward_std': rewards.std().item()
        }
        
        return metrics
    
    def _compute_rewards(self, sequences: torch.Tensor) -> torch.Tensor:
        """Compute rewards for generated sequences."""
        batch_size = sequences.shape[0]
        rewards = torch.zeros(batch_size, device=self.device)
        
        for i, sequence in enumerate(sequences):
            # Decode sequence to SMILES (simplified)
            smiles = self._decode_sequence(sequence)
            
            # Compute reward
            reward = self.reward_calculator.compute_reward(smiles)
            rewards[i] = reward
        
        return rewards
    
    def _decode_sequence(self, sequence: torch.Tensor) -> str:
        """Decode tensor sequence to SMILES string."""
        # Simplified decoding - in practice, use proper tokenizer
        # For now, return a dummy SMILES
        return "CCO"
    
    def _evaluate_generation(self) -> Dict[str, float]:
        """Evaluate the model's generation capabilities."""
        self.model.eval()
        
        # Generate molecules for evaluation
        num_molecules = 1000
        generated_molecules = []
        
        with torch.no_grad():
            for _ in range(num_molecules // 100):
                batch_molecules = self.model.generate_molecules(
                    batch_size=100,
                    max_length=self.config.model.generation_max_length,
                    temperature=0.8,
                    device=self.device
                )
                
                # Decode molecules
                for seq in batch_molecules['sequences']:
                    smiles = self._decode_sequence(seq[0] if isinstance(seq, list) else seq)
                    generated_molecules.append(smiles)
        
        # Evaluate with molecular evaluator
        evaluation_results = self.evaluator.evaluate(
            generated_molecules=generated_molecules,
            reference_molecules=self.reference_molecules,
            detailed_analysis=False
        )
        
        metrics = evaluation_results.get('metrics', {})
        
        # Compute average reward
        rewards = [self.reward_calculator.compute_reward(mol) for mol in generated_molecules]
        metrics['average_reward'] = np.mean(rewards)
        
        self.model.train()
        return metrics
    
    def _periodic_evaluation(self) -> None:
        """Perform periodic evaluation during training."""
        eval_metrics = self._evaluate_generation()
        
        # Log metrics
        self.logger.log_metrics(
            eval_metrics,
            step=self.global_step,
            prefix="stage2_eval"
        )
        
        # Generate and log sample molecules
        sample_molecules = self._generate_sample_molecules(10)
        self.logger.log_generated_molecules(
            sample_molecules,
            step=self.global_step,
            stage="stage2"
        )
    
    def _generate_sample_molecules(self, num_molecules: int) -> List[str]:
        """Generate sample molecules for logging."""
        self.model.eval()
        
        molecules = []
        with torch.no_grad():
            generated_data = self.model.generate_molecules(
                batch_size=num_molecules,
                max_length=self.config.model.generation_max_length,
                temperature=0.8,
                device=self.device
            )
            
            for seq in generated_data['sequences']:
                smiles = self._decode_sequence(seq[0] if isinstance(seq, list) else seq)
                molecules.append(smiles)
        
        self.model.train()
        return molecules
    
    def _generate_final_molecules(self) -> List[str]:
        """Generate final molecules after training completion."""
        print("Generating final molecules...")
        
        self.model.eval()
        final_molecules = []
        
        with torch.no_grad():
            for _ in range(10):  # Generate 1000 molecules
                batch_molecules = self.model.generate_molecules(
                    batch_size=100,
                    max_length=self.config.model.generation_max_length,
                    temperature=0.8,
                    device=self.device
                )
                
                for seq in batch_molecules['sequences']:
                    smiles = self._decode_sequence(seq[0] if isinstance(seq, list) else seq)
                    final_molecules.append(smiles)
        
        print(f"Generated {len(final_molecules)} final molecules")
        return final_molecules
    
    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save training checkpoint."""
        additional_state = {
            'stage': 2,
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_reward': self.best_reward,
            'reward_history': list(self.reward_history),
            'loss_history': list(self.loss_history)
        }
        
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            step=self.global_step,
            loss=-self.best_reward,  # Use negative reward as loss
            metrics={'best_reward': self.best_reward},
            is_best=is_best,
            additional_state=additional_state
        )


class RewardCalculator:
    """Calculates rewards for generated molecules."""
    
    def __init__(self):
        """Initialize reward calculator."""
        self.validator = SMILESValidator()
        self.property_calc = PropertyCalculator()
        
        # Reward weights
        self.weights = {
            'validity': 1.0,
            'qed': 0.5,
            'drug_likeness': 0.3,
            'novelty': 0.2
        }
    
    def compute_reward(self, smiles: str) -> float:
        """Compute reward for a SMILES string."""
        total_reward = 0.0
        
        # Validity reward
        if self.validator.is_valid(smiles):
            total_reward += self.weights['validity']
            
            # Property-based rewards
            qed = self.property_calc.calculate_qed(smiles)
            if qed is not None:
                total_reward += self.weights['qed'] * qed
            
            # Drug-likeness reward
            props = self.property_calc.calculate_all_properties(smiles)
            if props.get('drug_like', False):
                total_reward += self.weights['drug_likeness']
        
        return total_reward


class ExperienceBuffer:
    """Buffer for storing RL experiences."""
    
    def __init__(self, max_size: int = 10000):
        """Initialize experience buffer."""
        self.max_size = max_size
        self.experiences = []
    
    def add_experience(self, sequences: torch.Tensor, rewards: torch.Tensor, values: torch.Tensor) -> None:
        """Add experience to buffer."""
        for i in range(len(sequences)):
            experience = {
                'sequence': sequences[i].cpu(),
                'reward': rewards[i].cpu(),
                'value': values[i].cpu()
            }
            
            self.experiences.append(experience)
            
            # Remove oldest experiences if buffer is full
            if len(self.experiences) > self.max_size:
                self.experiences.pop(0)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch from experience buffer."""
        if len(self.experiences) < batch_size:
            indices = range(len(self.experiences))
        else:
            indices = np.random.choice(len(self.experiences), batch_size, replace=False)
        
        batch = {
            'sequences': [],
            'rewards': [],
            'values': []
        }
        
        for idx in indices:
            exp = self.experiences[idx]
            batch['sequences'].append(exp['sequence'])
            batch['rewards'].append(exp['reward'])
            batch['values'].append(exp['value'])
        
        return {
            'sequences': torch.stack(batch['sequences']),
            'rewards': torch.stack(batch['rewards']),
            'values': torch.stack(batch['values'])
        }
    
    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.experiences)