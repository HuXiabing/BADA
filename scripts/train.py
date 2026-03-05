import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
from config import get_config
from data import get_dataloader
from models import get_model
from trainers import Trainer
from utils import ExperimentManager

def main():
    parser = argparse.ArgumentParser(description="RISC-V Instruction Throughput Prediction Model Training")
    # Data arguments
    parser.add_argument("--train_data", type=str, default="data/train_data.json", help="Path to training data")
    parser.add_argument("--val_data", type=str, default="data/val_data.json", help="Path to validation data")
    # Model arguments
    parser.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "lstm", "gnn"], help="Model type")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--output_dir", type=str, default="../experiments", help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="default", help="Experiment name")

    args = parser.parse_args()
    experiment_manager = ExperimentManager(args.experiment_name, args.output_dir)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", args.device)

    config = get_config(
        model_type=args.model_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=experiment_manager.experiment_dir,
        experiment_name=args.experiment_name,
    )

    experiment_manager.save_config(config)

    train_loader = get_dataloader(
        config.model_type,
        dataset_path = config.train_data_path,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory = True,
        enable_chunking=False
    )

    val_loader = get_dataloader(
        config.model_type,
        dataset_path = config.val_data_path,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        enable_chunking=False
    )

    model = get_model(config)
    print(f"Create model: {config.model_type.upper()}, Number of parameters: {model.count_parameters():,}")

    trainer = Trainer(model, config, experiment_manager.experiment_dir, experiment_manager)

    experiment_manager.start(config.train_data_path, config.val_data_path, train_loader.dataset, val_loader.dataset)
    history = trainer.train(train_loader, val_loader)
    # return {
    #     "train_losses": self.train_losses,
    #     "val_losses": self.val_losses,
    #     "learning_rates": self.learning_rates,
    #     "best_metric": self.best_metric,
    #     "best_epoch": self.best_epoch
    # }

    experiment_manager.history = history
    experiment_manager.save_history()

    experiment_manager.save_summary({
        'model_type': config.model_type,
        'best_val_loss': history['best_metric'],
        'best_epoch': history['best_epoch'],
        'parameters': model.count_parameters(),
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset)
    })

    experiment_manager.finish()
    print(f"Finished! Results saved to: {experiment_manager.experiment_dir}")


if __name__ == "__main__":
    main()