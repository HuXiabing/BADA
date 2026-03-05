import os
import sys
import argparse
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
from utils import set_seed

def main():
    parser = argparse.ArgumentParser(
        description="RISC-V throughput prediction framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Select a command to execute")

    # Training command
    train_parser = subparsers.add_parser("train", help="Training RISC-V throughput prediction model")
    train_parser.add_argument("--model_type", type=str, default="transformer",
                              choices=["transformer", "gnn", "lstm"], help="Model type")
    train_parser.add_argument("--train_data", type=str, default="data/train_data.json", help="Path to training data")
    train_parser.add_argument("--val_data", type=str, default="data/val_data.json", help="Path to validation data")

    train_parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    train_parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    train_parser.add_argument("--output_dir", type=str, default="../experiments", help="Output directory")
    train_parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
    train_parser.add_argument("--device", type=str, default=None, help="Device to run on, defaults to automatic selection.")
    
    # resume training command
    resume_parser = subparsers.add_parser("resume", help="resume training from a checkpoint")
    resume_parser.add_argument("--train_data", type=str, default="data/train_data.json", help="Path to training data")
    resume_parser.add_argument("--val_data", type=str, default="data/val_data.json", help="Path to validation data")
    resume_parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint path")
    resume_parser.add_argument("--additional_epochs", type=int, default=50, help="extra epochs")
    resume_parser.add_argument("--experiment_name", type=str, default=None, help="new experiment name")

    # Incremental Learning Command
    incremental_parser = subparsers.add_parser("incremental",
                                               help="Perform incremental learning based on an existing model")
    incremental_parser.add_argument("--model_path", type=str, default=None,
                                    help="Path to the checkpoint of the trained model. Defaults to the most recently trained model.")
    incremental_parser.add_argument("--original", type=str, default=None,
                                    help="Original experiment directory, used to automatically find the latest model (if model_path is not specified).")
    incremental_parser.add_argument("--train_data", type=str, default="data/train_data.json",
                                    help="Path to the newly generated training data.")
    incremental_parser.add_argument("--val_data", type=str, default="data/val_data.json",
                                    help="Path to the validation data.")
    incremental_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for incremental training.")
    incremental_parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    incremental_parser.add_argument("--lr", type=float, default=3e-6,
                                    help="Learning rate, typically smaller than the initial learning rate.")
    incremental_parser.add_argument("--output_dir", type=str, default="../experiments",
                                    help="Output directory.")
    incremental_parser.add_argument("--experiment_name", type=str, default=None,
                                    help="Experiment name, defaults to incremental_{original_model_name}.")
    incremental_parser.add_argument("--restart_optimizer", action="store_true",
                                    help="Whether to reinitialize the optimizer (defaults to using the optimizer state from the original model).")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")


    args = parser.parse_args()
    set_seed(args.seed)
    
    # Get the current time as the default experiment name
    if args.command in ["train"] and args.experiment_name is None:
        args.experiment_name = f"{args.model_type}"
    if args.command in ["incremental"] and args.experiment_name is None:
        args.experiment_name = "incremental"
    
    if args.command == "train":
        from scripts.train import main as train_main
        sys.argv = [sys.argv[0]] + [
            "--model_type", args.model_type,
            "--train_data", args.train_data,
            "--val_data", args.val_data,
            "--batch_size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--output_dir", args.output_dir,
            "--experiment_name", args.experiment_name,
        ]
        train_main()

    elif args.command == "resume":
        from scripts.resume_training import main as resume_main
        sys.argv = [sys.argv[0]] + [
            "--checkpoint", args.checkpoint,
            "--train_data", args.train_data,
            "--val_data", args.val_data,
            "--additional_epochs", str(args.additional_epochs),
            "--seed", str(args.seed)
        ]

        if args.experiment_name:
            sys.argv.extend(["--experiment_name", args.experiment_name])

        if args.device:
            sys.argv.extend(["--device", args.device])
        resume_main()

    elif args.command == "incremental":
        from scripts.incremental_training import main as incremental_main
        sys.argv = [sys.argv[0]] + [
            "--train_data", args.train_data,
            "--val_data", args.val_data,
            "--batch_size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--output_dir", args.output_dir,
            "--seed", str(args.seed)
        ]

        if args.model_path:
            sys.argv.extend(["--model_path", args.model_path])
        if args.original:
            sys.argv.extend(["--original", args.original])
        if args.experiment_name:
            sys.argv.extend(["--experiment_name", args.experiment_name])
        if args.restart_optimizer:
            sys.argv.append("--restart_optimizer")

        incremental_main()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
    
