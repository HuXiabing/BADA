import sys
from pathlib import Path
import torch
import os
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config
from data import get_dataloader
from models import get_model
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import json
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau
from utils.metrics import MapeLoss, compute_accuracy
import argparse

class ChunkedSampleAggregator:

    def __init__(self):
        self.reset()

    def reset(self):
        # {original_idx: {'chunks': [], 'target': float, 'sample_idx': int}}
        self.chunked_predictions = defaultdict(lambda: {'chunks': [], 'target': None, 'sample_idx': None})
        # [(prediction, target, sample_idx)]
        self.non_chunked_predictions = []

    def is_chunked_sample(self, chunk_info):
        return chunk_info.get('is_chunked', False)

    def add_prediction(self, prediction, target, sample_idx, chunk_info):
        if self.is_chunked_sample(chunk_info):
            original_idx = chunk_info.get('original_idx', sample_idx)
            self.chunked_predictions[original_idx]['chunks'].append({
                'chunk_id': chunk_info['chunk_id'],
                'prediction': prediction,
                'total_chunks': chunk_info['total_chunks'],
                'window_size': chunk_info['window_size']
            })
            self.chunked_predictions[original_idx]['target'] = target
            self.chunked_predictions[original_idx]['sample_idx'] = sample_idx
        else:
            self.non_chunked_predictions.append((prediction, target, sample_idx))

    def get_aggregated_results(self):
        final_predictions = []
        final_targets = []
        final_indices = []

        for pred, target, idx in self.non_chunked_predictions:
            final_predictions.append(pred)
            final_targets.append(target)
            final_indices.append(idx)

        for original_idx, data in self.chunked_predictions.items():
            chunks = data['chunks']
            if chunks[0]['window_size'] == 0:
                chunks_sorted = sorted(chunks, key=lambda x: x['chunk_id'])
                aggregated_prediction = sum(chunk['prediction'] for chunk in chunks_sorted)
                final_predictions.append(aggregated_prediction)
                final_targets.append(data['target'])
                final_indices.append(data['sample_idx'])
            else:
                chunks_sorted = sorted(chunks, key=lambda x: x['chunk_id'])
                aggregated_prediction = sum(chunk['prediction'] for chunk in chunks_sorted) / chunks[0]['total_chunks']
                final_predictions.append(aggregated_prediction)
                final_targets.append(data['target'])
                final_indices.append(data['sample_idx'])

        return final_predictions, final_targets, final_indices

def test(test_loader, model, criterion, device='cuda', model_type='lstm'):
    model.eval()
    aggregator = ChunkedSampleAggregator()
    progress_bar = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            x = batch['X'].to(device)
            y = batch['Y'].to(device)
            idx = batch['idx']
            output = model(x)

            if model_type in ['lstm', 'transformer']:
                chunk_infos = batch['chunk_info']
                for i in range(len(output)):
                    aggregator.add_prediction(
                        prediction=output[i].item(),
                        target=y[i].item(),
                        sample_idx=idx[i].item(),
                        chunk_info=chunk_infos[i]
                    )
            elif model_type == 'gnn':
                chunk_infos = [json.loads(info_str) for info_str in batch['chunk_info']]
                graph_list = batch['X'].to_data_list()
                for i, graph in enumerate(graph_list):
                    aggregator.add_prediction(
                        prediction=output[i].item(),
                        target=y[i].item(),
                        sample_idx=idx[i].item(),
                        chunk_info=chunk_infos[i]
                    )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

    # aggregated_predictions, aggregated_targets, aggregated_indices = aggregator.get_aggregated_results()
    return aggregator.get_aggregated_results()

def inference(data_loader, model, device='cuda', model_type='gnn'):

    model.eval()
    aggregator = ChunkedSampleAggregator()

    with torch.no_grad():
        for batch in data_loader:
            x = batch['X'].to(device)
            idx = batch['idx']
            output = model(x)

            if model_type in ['lstm', 'transformer']:
                chunk_infos = batch['chunk_info']
                for i in range(len(output)):
                    aggregator.add_prediction(
                        prediction=output[i].item(),
                        target=0.0,
                        sample_idx=idx[i].item(),
                        chunk_info=chunk_infos[i]
                    )

            elif model_type == 'gnn':
                chunk_infos = [json.loads(info_str) for info_str in batch['chunk_info']]
                graph_list = batch['X'].to_data_list()
                for i, graph in enumerate(graph_list):
                    aggregator.add_prediction(
                        prediction=output[i].item(),
                        target=0.0,
                        sample_idx=idx[i].item(),
                        chunk_info=chunk_infos[i]
                    )

            else:
                raise ValueError(f"Unknown model type: {model_type}")

    predictions, _, indices = aggregator.get_aggregated_results()
    return predictions, indices

def main_inference():
    predictions = inference(data_loader, model, device, config.model_type)

def main_test():
    epsilon = getattr(config, 'loss_epsilon', 1e-5)
    criterion = MapeLoss(epsilon=epsilon)

    pred, true, aggregated_indices = test(
        test_loader=data_loader,
        model=model,
        criterion=criterion,
        device='cuda',
        model_type=config.model_type,
    )
    pred_tensor = torch.tensor(pred, dtype=torch.float32).to("cuda")
    true_tensor = torch.tensor(true, dtype=torch.float32).to("cuda")

    loss = criterion(pred_tensor, true_tensor)
    max_index = loss.argmax().item()
    metrics = {"loss": (loss.sum() / len(pred)).item()}

    pearsonr_corr, pearsonr_p_value = pearsonr(pred, true)
    spearmanr_corr, spearmanr_p_value = spearmanr(pred, true)
    kendalltau_corr, kendalltau_p_value = kendalltau(pred, true)

    current_accuracy5 = compute_accuracy(true, pred, 5)
    current_accuracy10 = compute_accuracy(true, pred, 10)
    current_accuracy25 = compute_accuracy(true, pred, 25)

    metrics.update({
        "pearsonr_corr": float(pearsonr_corr),
        "pearsonr_p_value": float(pearsonr_p_value),
        "spearmanr_corr": float(spearmanr_corr),
        "spearmanr_p_value": float(spearmanr_p_value),
        "kendalltau_corr": float(kendalltau_corr),
        "kendalltau_p_value": float(kendalltau_p_value),
        "accuracy5": current_accuracy5,
        "accuracy10": current_accuracy10,
        "accuracy25": current_accuracy25
    })

    print(f"\nTest Results:")
    print(f"  Loss: {metrics['loss']:.6f}")
    print(f"  Accuracy: {current_accuracy25:.6f}")
    print(f"  Pearson Correlation: {pearsonr_corr:.6f}")
    print(f"  Spearman Correlation: {spearmanr_corr:.6f}")
    print(f"  Kendall Tau Correlation: {kendalltau_corr:.6f}")
    
    result_dir = f"../experiments/{args.file}"
    with open(f"{result_dir}/test_result.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Results saved successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None, help="Path to training data")
    parser.add_argument("--test_data", type=str, default="datasets/u74.json", help="Path to test data")
    args = parser.parse_args()

    checkpoint_path = Path(f'../experiments/{args.file}/checkpoints/model_best.pth')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = Config(**checkpoint.get('config', {}))

    device = torch.device(config.device)
    data_loader = get_dataloader(
        config.model_type,
        args.test_data,
        enable_chunking=True,
        window_size=0,
        step_size=0,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = get_model(config).to(device)
    model.load_state_dict(checkpoint['model_state'])

    main_test()
    # main_inference()