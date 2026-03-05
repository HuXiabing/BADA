from .tokenizer import RISCVTokenizer
import torch
from .gnn_dataset import *
from .dataset import *
from torch_geometric.loader import DataLoader as PyGDataLoader

__all__ = ["get_dataloader",
           "RISCVGraphDataset"]

def get_dataloader(model_type, dataset_path: str,
                  batch_size: int = 32,
                  enable_chunking=True,
                  shuffle: bool = True,
                  num_workers: int = 0,
                  window_size=0,
                  step_size=0,
                  pin_memory=True) -> torch.utils.data.DataLoader:
    """
    Create a data loader

    Args:
        dataset_path: Path to the dataset json file
        batch_size: Batch size
        shuffle: Whether to shuffle the data

    Returns:
        Data loader
    """

    if model_type.lower() == "gnn":
        dataset = RISCVGraphDataset(dataset_path,
                                    cache_dir="./cache",
                                    rebuild_cache=False,
                                    enable_chunking=enable_chunking,
                                    window_size=window_size,
                                    step_size=step_size)
        return PyGDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory and torch.cuda.is_available()
        )

    elif model_type.lower() == "transformer":
        dataset = DatasetWithDistanceWeight(dataset_path,
                                            enable_chunking=enable_chunking,
                                            window_size=window_size,
                                            step_size=step_size)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn_transformer,
            shuffle=shuffle,
            num_workers=num_workers
        )

    elif model_type.lower() == "lstm":
        dataset = RNNDataset(dataset_path,
                             enable_chunking=enable_chunking,
                             window_size=window_size,
                             step_size=step_size)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn_lstm,
            shuffle=shuffle,
            num_workers=num_workers
        )

    else:
        raise ValueError(f"Model type {model_type} not supported")