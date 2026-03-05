import os
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
from config import Config
from data import get_dataloader
from models import get_model
from trainers import Trainer
from utils import set_seed, ExperimentManager


def main():
    parser = argparse.ArgumentParser(description="resume training from a checkpoint")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint path")
    parser.add_argument("--train_data", type=str, default="data/train_data.json", help="Path to training data(HDF5)")
    parser.add_argument("--val_data", type=str, default="data/val_data.json", help="Path to validation data(HDF5)")
    parser.add_argument("--additional_epochs", type=int, default=10, help="extra training epochs")
    parser.add_argument("--experiment_name", type=str, help="new experiment name")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Device to run on.")
    
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    config_dict = checkpoint.get('config', None)
    if config_dict is None:
        raise ValueError(f"There is no configuration information in {checkpoint_path}")
    
    config = Config(**config_dict)
    
    if args.device:
        config.device = args.device
    device = torch.device(config.device)

    if args.experiment_name:
        new_experiment_name = args.experiment_name
    else:
        old_experiment_name = config.experiment_name
        new_experiment_name = f"{old_experiment_name}_continued"

    total_epochs = checkpoint.get('epoch', 0) + 1 + args.additional_epochs
    experiment_manager = ExperimentManager(new_experiment_name, config.output_dir)
    
    # update config
    config.experiment_name = new_experiment_name
    config.experiment_dir = experiment_manager.experiment_dir
    config.checkpoint_dir = experiment_manager.checkpoint_dir
    config.log_dir = experiment_manager.log_dir
    config.epochs = total_epochs
    experiment_manager.save_config(config)
    
    train_loader = get_dataloader(
        config.model_type,
        args.train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = get_dataloader(
        config.model_type,
        args.val_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    model = get_model(config)

    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    
    trainer = Trainer(model, config, experiment_manager.experiment_dir,experiment_manager)
    
    if hasattr(trainer, 'optimizer') and 'optimizer_state' in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
        print("Loaded the optimizer state from the previous model")
    
    if hasattr(trainer, 'scheduler') and 'scheduler_state' in checkpoint:
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state'])
        print("Loaded the scheduler state from the previous model")
    
    trainer.start_epoch = checkpoint.get('epoch', 0) + 1
    trainer.global_step = checkpoint.get('global_step', 0)
    trainer.best_metric = checkpoint.get('best_metric', float('inf'))
    trainer.best_epoch = checkpoint.get('best_epoch', 0)
    trainer.train_losses = checkpoint.get('train_losses', [])
    trainer.val_losses = checkpoint.get('val_losses', [])
    trainer.learning_rates = checkpoint.get('learning_rates', [])

    print(f"Resuming training from checkpoint {args.checkpoint}")
    print(f"Original number of epochs: {trainer.start_epoch}, additional epochs: {args.additional_epochs}")
    print(f"Current best validation loss: {trainer.best_metric:.6f} at Epoch {trainer.best_epoch + 1}")
    print(f"Continue training, device: {device}")

    history = trainer.train(train_loader, val_loader)
    experiment_manager.history = history
    experiment_manager.save_history()

    experiment_manager.save_summary({
        'model_type': config.model_type,
        'best_val_loss': history['best_metric'],
        'best_epoch': history['best_epoch'] + 1,
        'parameters': model.count_parameters(),
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'original_checkpoint': str(checkpoint_path),
        'additional_epochs': args.additional_epochs,
        'resume_training': True
    })
    
    experiment_manager.finish()
    print(f"Resume training completed! Model saved at: {experiment_manager.experiment_dir}")
    print(f"Best validation loss: {history['best_metric']:.6f} at Epoch {history['best_epoch'] + 1}")


if __name__ == "__main__":
    main()
