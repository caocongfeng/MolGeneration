"""Stage 1 trainer for chemical reasoning pre-training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from typing import Dict, List, Optional, Any
import numpy as np
from tqdm import tqdm
import logging

from molgeneration.config.config import MolGenerationConfig
from molgeneration.models.reasoning_model import ReasoningModel
from molgeneration.data.loader import DataLoader
from molgeneration.utils.training_utils import CheckpointManager, LoggingUtils

logger = logging.getLogger(__name__)


class Stage1Trainer:
    """
    Trainer for Stage 1: Chemical reasoning pre-training.
    
    Trains the model to understand chemical concepts, properties,
    and relationships before molecule generation.
    """
    
    def __init__(
        self,
        model: ReasoningModel,
        config: MolGenerationConfig,
        data_loader: DataLoader,
        device: torch.device,
        checkpoint_manager: CheckpointManager,
        logger: LoggingUtils
    ):
        """
        Initialize Stage 1 trainer.
        
        Args:
            model: Reasoning model to train
            config: Training configuration
            data_loader: Data loader for reasoning data
            device: Training device
            checkpoint_manager: Checkpoint manager
            logger: Logging utilities
        """
        self.model = model
        self.config = config
        self.data_loader = data_loader
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Loss weights for multi-task learning
        self.loss_weights = {
            'language_modeling': 1.0,
            'property_prediction': 0.5,
            'similarity_reasoning': 0.3,
            'functional_group': 0.3,
            'reaction_prediction': 0.2,
            'task_classification': 0.1,
        }
        
        print("Stage 1 trainer initialized")
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer for Stage 1 training."""
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.stage1_lr,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        return optimizer
    
    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        total_steps = self.config.training.stage1_epochs * 1000  # Approximate
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
        Execute Stage 1 training.
        
        Returns:
            Training results and metrics
        """
        print(f"Starting Stage 1 training for {self.config.training.stage1_epochs} epochs...")
        
        self.model.train()
        
        training_results = {
            'epochs_completed': 0,
            'best_loss': float('inf'),
            'final_metrics': {},
            'training_history': []
        }
        
        for epoch in range(self.config.training.stage1_epochs):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch + 1}/{self.config.training.stage1_epochs}")
            
            # Training step
            epoch_metrics = self._train_epoch()
            
            # Validation step
            val_metrics = self._validate_epoch()
            
            # Combine metrics
            combined_metrics = {
                **{f"train_{k}": v for k, v in epoch_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()}
            }
            
            # Log metrics
            self.logger.log_metrics(
                combined_metrics,
                step=self.global_step,
                epoch=epoch,
                prefix="stage1"
            )
            
            # Save checkpoint if best model
            is_best = val_metrics['total_loss'] < self.best_loss
            if is_best:
                self.best_loss = val_metrics['total_loss']
                self._save_checkpoint(is_best=True)
            
            # Regular checkpoint
            if (epoch + 1) % self.config.training.save_steps == 0:
                self._save_checkpoint(is_best=False)
            
            # Update training history
            training_results['training_history'].append({
                'epoch': epoch,
                'train_loss': epoch_metrics['total_loss'],
                'val_loss': val_metrics['total_loss'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            training_results['epochs_completed'] = epoch + 1
        
        training_results['best_loss'] = self.best_loss
        training_results['final_metrics'] = combined_metrics
        
        print(f"Stage 1 training completed. Best loss: {self.best_loss:.4f}")
        
        return training_results
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {
            'total_loss': [],
            'lm_loss': [],
            'property_loss': [],
            'similarity_loss': [],
            'functional_group_loss': [],
            'reaction_loss': [],
            'task_classification_loss': []
        }
        
        # Create reasoning data loader
        reasoning_dataloader = self.data_loader.create_reasoning_dataloader(
            data_path="data/reasoning/property_prediction.json",  # Placeholder
            batch_size=self.config.training.batch_size,
            shuffle=True,
            split="train"
        )
        
        progress_bar = tqdm(reasoning_dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch.get('labels', batch['input_ids']),  # Use input as labels for LM
                return_reasoning_outputs=True
            )
            
            # Compute losses
            losses = self._compute_losses(outputs, batch)
            total_loss = losses['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.training.max_grad_norm
            )
            
            self.optimizer.step()
            self.global_step += 1
            
            # Accumulate losses
            for loss_name, loss_value in losses.items():
                if loss_name in epoch_losses:
                    epoch_losses[loss_name].append(loss_value.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log periodically
            if self.global_step % self.config.training.logging_steps == 0:
                step_metrics = {k: v.item() for k, v in losses.items()}
                self.logger.log_metrics(
                    step_metrics,
                    step=self.global_step,
                    prefix="stage1_step"
                )
        
        # Average losses for epoch
        epoch_metrics = {
            loss_name: np.mean(loss_values) 
            for loss_name, loss_values in epoch_losses.items()
            if loss_values
        }
        
        return epoch_metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        val_losses = {
            'total_loss': [],
            'lm_loss': [],
            'property_loss': [],
            'similarity_loss': [],
            'functional_group_loss': [],
            'reaction_loss': [],
            'task_classification_loss': []
        }
        
        # Create validation data loader
        val_dataloader = self.data_loader.create_reasoning_dataloader(
            data_path="data/reasoning/property_prediction.json",  # Placeholder
            batch_size=self.config.training.batch_size,
            shuffle=False,
            split="val"
        )
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch.get('labels', batch['input_ids']),
                    return_reasoning_outputs=True
                )
                
                # Compute losses
                losses = self._compute_losses(outputs, batch)
                
                # Accumulate losses
                for loss_name, loss_value in losses.items():
                    if loss_name in val_losses:
                        val_losses[loss_name].append(loss_value.item())
        
        # Average losses
        val_metrics = {
            loss_name: np.mean(loss_values) 
            for loss_name, loss_values in val_losses.items()
            if loss_values
        }
        
        self.model.train()
        return val_metrics
    
    def _compute_losses(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute multi-task losses for reasoning training."""
        losses = {}
        
        # Language modeling loss
        if 'lm_loss' in outputs:
            losses['lm_loss'] = outputs['lm_loss']
        else:
            losses['lm_loss'] = torch.tensor(0.0, device=self.device)
        
        # Task classification loss
        if 'task_logits' in outputs:
            # Create dummy task labels (in practice, these would come from the batch)
            batch_size = outputs['task_logits'].shape[0]
            task_labels = torch.randint(0, outputs['task_logits'].shape[1], (batch_size,), device=self.device)
            losses['task_classification_loss'] = F.cross_entropy(outputs['task_logits'], task_labels)
        else:
            losses['task_classification_loss'] = torch.tensor(0.0, device=self.device)
        
        # Property prediction loss
        if 'all_task_outputs' in outputs and 'property_prediction' in outputs['all_task_outputs']:
            prop_outputs = outputs['all_task_outputs']['property_prediction']
            if 'property_predictions' in prop_outputs:
                # Create dummy property targets
                batch_size = prop_outputs['property_predictions'].shape[0]
                prop_targets = torch.randn_like(prop_outputs['property_predictions'])
                losses['property_loss'] = F.mse_loss(prop_outputs['property_predictions'], prop_targets)
            else:
                losses['property_loss'] = torch.tensor(0.0, device=self.device)
        else:
            losses['property_loss'] = torch.tensor(0.0, device=self.device)
        
        # Similarity reasoning loss
        if 'all_task_outputs' in outputs and 'similarity_reasoning' in outputs['all_task_outputs']:
            sim_outputs = outputs['all_task_outputs']['similarity_reasoning']
            if 'similarity_logits' in sim_outputs:
                # Create dummy similarity labels
                batch_size = sim_outputs['similarity_logits'].shape[0]
                sim_labels = torch.randint(0, sim_outputs['similarity_logits'].shape[1], (batch_size,), device=self.device)
                losses['similarity_loss'] = F.cross_entropy(sim_outputs['similarity_logits'], sim_labels)
            else:
                losses['similarity_loss'] = torch.tensor(0.0, device=self.device)
        else:
            losses['similarity_loss'] = torch.tensor(0.0, device=self.device)
        
        # Functional group loss
        if 'all_task_outputs' in outputs and 'functional_group' in outputs['all_task_outputs']:
            fg_outputs = outputs['all_task_outputs']['functional_group']
            if 'functional_group_logits' in fg_outputs:
                # Create dummy functional group labels (multi-label)
                batch_size = fg_outputs['functional_group_logits'].shape[0]
                num_groups = fg_outputs['functional_group_logits'].shape[1]
                fg_labels = torch.randint(0, 2, (batch_size, num_groups), device=self.device).float()
                losses['functional_group_loss'] = F.binary_cross_entropy_with_logits(
                    fg_outputs['functional_group_logits'], fg_labels
                )
            else:
                losses['functional_group_loss'] = torch.tensor(0.0, device=self.device)
        else:
            losses['functional_group_loss'] = torch.tensor(0.0, device=self.device)
        
        # Reaction prediction loss
        if 'all_task_outputs' in outputs and 'reaction_prediction' in outputs['all_task_outputs']:
            reaction_outputs = outputs['all_task_outputs']['reaction_prediction']
            if 'reaction_logits' in reaction_outputs:
                # Create dummy reaction labels
                batch_size = reaction_outputs['reaction_logits'].shape[0]
                reaction_labels = torch.randint(0, reaction_outputs['reaction_logits'].shape[1], (batch_size,), device=self.device)
                losses['reaction_loss'] = F.cross_entropy(reaction_outputs['reaction_logits'], reaction_labels)
            else:
                losses['reaction_loss'] = torch.tensor(0.0, device=self.device)
        else:
            losses['reaction_loss'] = torch.tensor(0.0, device=self.device)
        
        # Combine losses with weights
        total_loss = torch.tensor(0.0, device=self.device)
        for loss_name, loss_value in losses.items():
            weight_name = loss_name.replace('_loss', '').replace('lm', 'language_modeling')
            weight = self.loss_weights.get(weight_name, 1.0)
            total_loss += weight * loss_value
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save training checkpoint."""
        additional_state = {
            'stage': 1,
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'loss_weights': self.loss_weights
        }
        
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            step=self.global_step,
            loss=self.best_loss,
            metrics={'best_loss': self.best_loss},
            is_best=is_best,
            additional_state=additional_state
        )
    
    def generate_reasoning_examples(self, num_examples: int = 5) -> List[str]:
        """Generate reasoning examples to monitor training progress."""
        self.model.eval()
        
        generated_examples = []
        
        with torch.no_grad():
            for _ in range(num_examples):
                # Create input prompt
                input_prompt = "Predict the molecular weight of molecule CCO."
                
                # Tokenize (simplified - in practice use proper tokenizer)
                input_ids = torch.tensor([[2, 3, 4, 5]], device=self.device)  # Dummy tokens
                
                # Generate
                generated = self.model.generate_reasoning(
                    input_ids=input_ids,
                    max_length=100,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95
                )
                
                # Decode (simplified)
                generated_text = f"Generated reasoning example {len(generated_examples) + 1}"
                generated_examples.append(generated_text)
        
        self.model.train()
        return generated_examples
    
    def evaluate_reasoning_capabilities(self) -> Dict[str, float]:
        """Evaluate the model's reasoning capabilities."""
        self.model.eval()
        
        evaluation_metrics = {
            'property_prediction_accuracy': 0.0,
            'similarity_accuracy': 0.0,
            'functional_group_accuracy': 0.0,
            'reaction_accuracy': 0.0,
            'average_reasoning_score': 0.0
        }
        
        # In practice, this would run evaluation on specific reasoning tasks
        # For now, return dummy metrics
        
        self.model.train()
        return evaluation_metrics