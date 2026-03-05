import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

class ExperimentManager:

    def __init__(self, experiment_name: str, base_dir: str = "../experiments"):

        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{self.timestamp}"
        self.experiment_dir = os.path.join(self.base_dir, self.experiment_id)

        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.log_dir = os.path.join(self.experiment_dir, "logs")
        self.setup_directories()
        self.setup_logger()
        
        self.metrics = {}
        self.history = {}
        self.start_time = time.time()
        self.logger.info(f"Experiment created: {self.experiment_id}")
    
    def setup_directories(self):
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def setup_logger(self):

        self.logger = logging.getLogger(self.experiment_id)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:

            log_file = os.path.join(self.log_dir, "experiment.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def save_config(self, config):

        config_path = os.path.join(self.experiment_dir, "config.json")
        
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        else:
            config_dict = dict(config)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        self.logger.info(f"Configuration saved to {config_path}")

    def log_metrics(self, metrics: Dict[str, Any], epoch: int, prefix: str = ""):
        """
        Log training/validation metrics

        Args:
            metrics: Dictionary of metrics
            {
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            ...
            }
            epoch: Current epoch (e.g., epoch)
            prefix: Metric prefix (e.g., 'train_' or 'val_')
        """
        metrics_str_parts = []
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                # If the value is numeric, use the .6f format
                metrics_str_parts.append(f"{name}: {value:.6f}")
            else:
                metrics_str_parts.append(f"{name}: {value}")

        metrics_str = ", ".join(metrics_str_parts)

        self.logger.info(f"Epoch {epoch} - {prefix}metrics: {metrics_str}")

    def save_history(self):

        history_path = os.path.join(self.log_dir, "history.json")

        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def save_summary(self, summary_data: Dict[str, Any]):

        os.makedirs(self.experiment_dir, exist_ok=True)
        summary_path = os.path.join(self.experiment_dir, "summary.json")

        try:
            summary_data['duration'] = time.time() - self.start_time

            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=4)

            self.logger.info(f"Experiment summary saved to {summary_path}")
        except Exception as e:
            self.logger.error(f"Error saving experiment summary: {e}")
            raise
    
    def finish(self):
        duration = time.time() - self.start_time
        self.logger.info(f"Experiment completed. Best validation loss: {self.history['best_metric']:.6f} at Epoch "
              f"{self.history['best_epoch']}. Total time: {duration:.2f} seconds")

    def start(self,train_data, val_data, train_dataset, val_dataset):
        self.logger.info(f"Training data: {train_data}, Number of samples: {len(train_dataset)}")
        self.logger.info(f"Validation data: {val_data}, Number of samples: {len(val_dataset)}")

    def save_loss_stats(self, loss_stats, epoch):

        stats_dir = os.path.join(self.experiment_dir, "statistics")
        os.makedirs(stats_dir, exist_ok=True)

        stats_path = os.path.join(stats_dir, f"{loss_stats['prefix']}_loss_stats_epoch_{epoch}.csv")
        loss = loss_stats["sample_loss"]
        loss.to_csv(stats_path, index=False)
        os.chmod(stats_path, 0o444) # Read-only

        self.logger.info(f"Loss statistics saved to {stats_path}")
