"""Training utilities for model management and logging."""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from datetime import datetime
import wandb
from tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints and state saving/loading."""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        step: int,
        loss: float,
        metrics: Dict[str, float],
        is_best: bool = False,
        additional_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            epoch: Current epoch
            step: Current step
            loss: Current loss value
            metrics: Dictionary of metrics
            is_best: Whether this is the best checkpoint
            additional_state: Additional state to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_state = {
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'metrics': metrics,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'timestamp': datetime.now().isoformat(),
        }
        
        if additional_state:
            checkpoint_state.update(additional_state)
        
        # Save regular checkpoint
        filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint_state, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint_state, best_path)
            logger.info(f"New best checkpoint saved: {best_path}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: PyTorch model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load tensors to
            
        Returns:
            Checkpoint metadata (epoch, step, loss, metrics)
        """
        if device is None:
            device = torch.device('cpu')
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
            'loss': checkpoint.get('loss', float('inf')),
            'metrics': checkpoint.get('metrics', {}),
            'timestamp': checkpoint.get('timestamp', ''),
        }
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return metadata
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to the latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoints:
            return None
        
        # Sort by modification time
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    def get_best_checkpoint(self) -> Optional[str]:
        """
        Get path to the best checkpoint.
        
        Returns:
            Path to best checkpoint or None if it doesn't exist
        """
        best_path = self.checkpoint_dir / "best_checkpoint.pt"
        return str(best_path) if best_path.exists() else None
    
    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints, keeping only the most recent ones."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by modification time
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        
        # Remove oldest checkpoints
        for checkpoint in checkpoints[:-self.max_checkpoints]:
            checkpoint.unlink()
            logger.info(f"Removed old checkpoint: {checkpoint}")


class LoggingUtils:
    """Utilities for experiment logging and tracking."""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str,
        use_wandb: bool = True,
        use_tensorboard: bool = True,
        wandb_project: str = "mol-generation"
    ):
        """
        Initialize logging utilities.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for logs
            use_wandb: Whether to use Weights & Biases
            use_tensorboard: Whether to use TensorBoard
            wandb_project: W&B project name
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        # Initialize TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            tb_dir = self.log_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.tb_writer = SummaryWriter(tb_dir)
        
        # Initialize W&B
        if use_wandb:
            try:
                wandb.init(
                    project=wandb_project,
                    name=experiment_name,
                    dir=str(self.log_dir)
                )
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
                self.use_wandb = False
        
        # Setup file logging
        self._setup_file_logging()
    
    def _setup_file_logging(self) -> None:
        """Setup file logging configuration."""
        log_file = self.log_dir / f"{self.experiment_name}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.INFO)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        epoch: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """
        Log metrics to all configured loggers.
        
        Args:
            metrics: Dictionary of metric values
            step: Current training step
            epoch: Current epoch (optional)
            prefix: Prefix for metric names
        """
        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Log to TensorBoard
        if self.tb_writer:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(name, value, step)
        
        # Log to W&B
        if self.use_wandb:
            log_dict = dict(metrics)
            log_dict['step'] = step
            if epoch is not None:
                log_dict['epoch'] = epoch
            wandb.log(log_dict)
        
        # Log to file
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {step} - {metrics_str}")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        # Save config to file
        config_file = self.log_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # Log to W&B
        if self.use_wandb:
            wandb.config.update(config)
        
        logger.info("Configuration logged")
    
    def log_model_architecture(self, model: nn.Module) -> None:
        """
        Log model architecture information.
        
        Args:
            model: PyTorch model
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        arch_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_architecture': str(model),
        }
        
        # Save to file
        arch_file = self.log_dir / "model_architecture.txt"
        with open(arch_file, 'w') as f:
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n\n")
            f.write("Model architecture:\n")
            f.write(str(model))
        
        # Log to W&B
        if self.use_wandb:
            wandb.config.update({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
            })
        
        logger.info(f"Model architecture logged - Total params: {total_params:,}")
    
    def log_generated_molecules(
        self,
        molecules: List[str],
        step: int,
        stage: str = "",
        max_display: int = 20
    ) -> None:
        """
        Log generated molecules.
        
        Args:
            molecules: List of generated SMILES
            step: Current training step
            stage: Training stage (e.g., 'stage1', 'stage2')
            max_display: Maximum number of molecules to display
        """
        # Save all molecules to file
        molecules_file = self.log_dir / f"generated_molecules_step_{step}.txt"
        with open(molecules_file, 'w') as f:
            for mol in molecules:
                f.write(f"{mol}\n")
        
        # Log sample to W&B
        if self.use_wandb:
            sample_molecules = molecules[:max_display]
            table_data = [[i, mol] for i, mol in enumerate(sample_molecules)]
            table = wandb.Table(data=table_data, columns=["Index", "SMILES"])
            
            wandb.log({
                f"{stage}/generated_molecules": table,
                f"{stage}/num_generated": len(molecules)
            }, step=step)
        
        logger.info(f"Generated {len(molecules)} molecules at step {step}")
    
    def close(self) -> None:
        """Close all loggers."""
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.use_wandb:
            wandb.finish()
        
        logger.info("Logging utilities closed")


class ModelUtils:
    """Utilities for model management and manipulation."""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """
        Count model parameters.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }
    
    @staticmethod
    def freeze_parameters(model: nn.Module, layer_names: Optional[List[str]] = None) -> None:
        """
        Freeze model parameters.
        
        Args:
            model: PyTorch model
            layer_names: Specific layer names to freeze (if None, freeze all)
        """
        if layer_names is None:
            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False
        else:
            # Freeze specific layers
            for name, module in model.named_modules():
                if any(layer_name in name for layer_name in layer_names):
                    for param in module.parameters():
                        param.requires_grad = False
    
    @staticmethod
    def unfreeze_parameters(model: nn.Module, layer_names: Optional[List[str]] = None) -> None:
        """
        Unfreeze model parameters.
        
        Args:
            model: PyTorch model
            layer_names: Specific layer names to unfreeze (if None, unfreeze all)
        """
        if layer_names is None:
            # Unfreeze all parameters
            for param in model.parameters():
                param.requires_grad = True
        else:
            # Unfreeze specific layers
            for name, module in model.named_modules():
                if any(layer_name in name for layer_name in layer_names):
                    for param in module.parameters():
                        param.requires_grad = True
    
    @staticmethod
    def get_model_size_mb(model: nn.Module) -> float:
        """
        Get model size in megabytes.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size in MB
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb