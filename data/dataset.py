import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import Dict, Any, Union, Optional, Tuple
import json
from .tokenizer import RISCVTokenizer

class TorchDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to(self, device):
        for k, v in self.items():
            if hasattr(v, 'to'):
                self[k] = v.to(device)
        return self

def make_attention_weight(mask, is_continual_pad=True):
    sizes = (~mask).sum(dim=1)
    maximum_size = mask.size(1)

    all_masking = []

    for idx, s in enumerate(sizes):
        cur_mask = ~mask[idx]

        i, j = torch.meshgrid(
            torch.arange(s, device=mask.device), torch.arange(s, device=mask.device), indexing='ij'
        )

        if is_continual_pad:
            masking = F.pad((s - abs(i - j)) / s, (0, maximum_size - s, 0, maximum_size - s), value=0)
        else:
            tmp = torch.full((maximum_size, maximum_size), 0.0, device=mask.device)
            tmp[cur_mask] = F.pad((s - abs(i - j)) / s, (0, maximum_size - s), value=0)

            masking = torch.full((maximum_size, maximum_size), 0.0, device=mask.device)
            masking[:, cur_mask] = tmp[:, :s]

        all_masking.append(masking)

    all_masking = torch.stack(all_masking)

    return all_masking

class DatasetWithDistanceWeight(Dataset):
    def __init__(self, json_path,
                 max_instr_length=8,
                 max_instr_count=64,
                 return_bb_mask=True,
                 return_seq_mask=True,
                 return_op_mask=True,
                 enable_chunking=True,
                 window_size=0,
                 step_size=0
                 ):

        with open(json_path, 'r', encoding='utf-8') as file:
            self.original_data = json.load(file)

        self.pad_idx = 0
        self.return_bb_mask = return_bb_mask
        self.return_seq_mask = return_seq_mask
        self.return_op_mask = return_op_mask
        self.enable_chunking = enable_chunking
        self.window_size = window_size
        self.step_size = step_size

        self.max_instr_length = max_instr_length
        self.max_instr_count = max_instr_count

        self.tokenizer = RISCVTokenizer()
        self.sample_mapping = self._create_sample_mapping()

    def _create_sample_mapping(self):

        sample_mapping = []
        for original_idx, sample in enumerate(self.original_data):
            bb = sample['instructions']
            if not self.enable_chunking or len(bb) <= self.max_instr_count:
                sample_mapping.append({
                    'original_idx': original_idx,
                    'chunk_id': 0,
                    'total_chunks': 1,
                    'is_chunked': False
                })
            else:
                if self.window_size == 0:
                    num_chunks = (len(bb) + self.max_instr_count - 1) // self.max_instr_count
                    for chunk_id in range(num_chunks):
                        sample_mapping.append({
                            'original_idx': original_idx,
                            'chunk_id': chunk_id,
                            'total_chunks': num_chunks,
                            'is_chunked': True
                        })
                else:
                    def _mod(a, b):
                        return a % b or b

                    original_length = len(bb)
                    original_size = 3
                    pad_front = self.window_size - original_size
                    divided = original_length - original_size
                    mod = _mod(divided, self.step_size)
                    pad_back = self.window_size - mod
                    padded_length = pad_front + original_length + pad_back
                    num_windows = (padded_length - self.window_size) // self.step_size + 1

                    for chunk_id in range(num_windows):
                        sample_mapping.append({
                            'original_idx': original_idx,
                            'chunk_id': chunk_id,
                            'total_chunks': num_windows,
                            'is_chunked': True
                        })

        return sample_mapping

    def _process_single_sample(self, original_idx, chunk_id, is_chunked):

        sample = self.original_data[original_idx]
        bb = sample['instructions']
        encoded = [self.tokenizer.encode_instruction(i) for i in bb]

        if not is_chunked:

            processed_x = self._pad_single_sample(encoded)
            chunk_info = {
                'is_chunked': False,
                'chunk_id': 0,
                'total_chunks': 1,
                'original_length': len(encoded),
                'window_size': self.window_size,
                'step_size': self.step_size
            }
        else:
            if self.window_size == 0:
                chunks = self._create_chunks(encoded)
                chunk_data = chunks[chunk_id]
                processed_x = self._pad_single_sample(chunk_data)
                chunk_info = {
                    'is_chunked': True,
                    'chunk_id': chunk_id,
                    'total_chunks': len(chunks),
                    'original_length': len(encoded),
                    'window_size': self.window_size,
                    'step_size': self.step_size
                }
            else:
                chunks = self._create_sliding_windows(encoded)
                chunk_data = chunks[chunk_id]
                processed_x = self._pad_single_sample(chunk_data)
                chunk_info = {
                    'is_chunked': True,
                    'chunk_id': chunk_id,
                    'total_chunks': int(self.window_size / self.step_size),
                    'original_length': len(encoded),
                    'window_size': self.window_size,
                    'step_size': self.step_size
                }

        return {
            'x': processed_x,
            'y': sample.get('throughput'),
            'idx': sample['idx'],
            'original_idx': original_idx,
            'chunk_info': chunk_info
        }

    def _create_sliding_windows(self, encoded):
        def _mod(a, b):
            return a % b or b

        original_length = len(encoded)
        original_size = 3
        pad_front = self.window_size - original_size
        divided = original_length - original_size
        mod = _mod(divided, self.step_size)
        pad_back = self.window_size - mod

        padding_instr = [self.pad_idx] * self.max_instr_length
        padded_sequence = ([padding_instr] * pad_front +
                           encoded +
                           [padding_instr] * pad_back)

        windows = []
        padded_length = len(padded_sequence)

        for start in range(0, padded_length - self.window_size + 1, self.step_size):
            end = start + self.window_size
            window_sequence = padded_sequence[start:end]
            windows.append(window_sequence)

        return windows

    # def _create_chunks(self, encoded):
    #
    #     chunks = []
    #     start = 0
    #     while start < len(encoded):
    #         end = min(start + self.max_instr_count, len(encoded))
    #         chunks.append(encoded[start:end])
    #         start = end
    #     return chunks

    def _create_chunks(self, encoded):

        total_length = len(encoded)
        num_chunks = (total_length + self.max_instr_count - 1) // self.max_instr_count

        base_size = total_length // num_chunks
        remainder = total_length % num_chunks

        chunks = []
        start = 0

        for i in range(num_chunks):
            chunk_size = base_size + (1 if i < remainder else 0)
            end = start + chunk_size
            chunks.append(encoded[start:end])
            start = end

        return chunks

    def _pad_single_sample(self, encoded):

        if len(encoded) > self.max_instr_count:
            encoded = encoded[:self.max_instr_count]

        padded_encoded = [
            F.pad(torch.tensor(instr, dtype=torch.long), (0, self.max_instr_length - len(instr)),
                  value=self.pad_idx)
            for instr in encoded
        ]

        if len(padded_encoded) < self.max_instr_count:
            padding = torch.full((self.max_instr_count - len(padded_encoded), self.max_instr_length), self.pad_idx,
                                 dtype=torch.long)
            padded_encoded.extend(padding)

        return torch.stack(padded_encoded)

    def __len__(self):
        return len(self.sample_mapping)

    def __getitem__(self, index):

        mapping = self.sample_mapping[index]
        sample = self._process_single_sample(
            mapping['original_idx'],
            mapping['chunk_id'],
            mapping['is_chunked']
        )
        return sample['x'], sample['y'], sample['idx'], sample['chunk_info']

    @staticmethod
    def make_input(x, pad_idx=0, return_bb_mask=True, return_seq_mask=True, return_op_mask=True):
        x_dict = {
            'x': x,
        }
        batch_size, inst_size, seq_size = x.shape  # batch_size, max_instr_count, max_instr_length
        mask = x == pad_idx
        bb_mask = mask.view(batch_size, inst_size * seq_size)
        seq_mask = mask.view(batch_size * inst_size, seq_size)
        op_mask = mask.all(dim=2)

        if return_bb_mask:
            bb_attn_mod = make_attention_weight(bb_mask, is_continual_pad=False)
            x_dict['bb_attn_mod'] = bb_attn_mod

        if return_seq_mask:
            seq_attn_mod = make_attention_weight(seq_mask)
            x_dict['seq_attn_mod'] = seq_attn_mod

        if return_op_mask:
            op_attn_mod = make_attention_weight(op_mask)
            x_dict['op_attn_mod'] = op_attn_mod

        return TorchDict(**x_dict)

def collate_fn_transformer(batch):
    xs, ys, idx, chunk_infos = zip(*batch)

    max_inst_count = max(x.size(0) for x in xs)
    max_inst_length = xs[0].size(1)

    padded_xs = []
    for x in xs:
        if x.size(0) < max_inst_count:
            padding = torch.full((max_inst_count - x.size(0), max_inst_length), 0, dtype=torch.long)
            x = torch.cat([x, padding], dim=0)
        padded_xs.append(x)

    xs = torch.stack(padded_xs)
    # ys = torch.tensor(ys, dtype=torch.float)
    if all(y is not None for y in ys):
        ys = torch.tensor(ys, dtype=torch.float)

    idx = torch.tensor(idx, dtype=torch.long)

    if hasattr(batch[0], '__self__'):
        dataset = batch[0].__self__
    else:
        dataset = None

    if dataset and isinstance(dataset, DatasetWithDistanceWeight):
        x_dict = DatasetWithDistanceWeight.make_input(
            xs,
            pad_idx=dataset.pad_idx,
            return_bb_mask=dataset.return_bb_mask,
            return_seq_mask=dataset.return_seq_mask,
            return_op_mask=dataset.return_op_mask
        )
    else:
        x_dict = DatasetWithDistanceWeight.make_input(xs)

    return {
        'X': x_dict,
        'idx': idx,
        'Y': ys,
        'chunk_info': chunk_infos
    }

# lazy load
class RNNDataset(Dataset):
    def __init__(self, json_path,
                 max_instr_length=8,
                 max_instr_count=64,
                 enable_chunking=True,
                 window_size=0,
                 step_size=0
                 ):

        with open(json_path, 'r', encoding='utf-8') as file:
            self.original_data = json.load(file)

        self.pad_idx = 0
        self.max_instr_length = max_instr_length
        self.max_instr_count = max_instr_count
        self.enable_chunking = enable_chunking
        self.window_size = window_size
        self.step_size = step_size

        self.tokenizer = RISCVTokenizer()
        self.sample_mapping = self._create_sample_mapping()

    def _create_sample_mapping(self):

        sample_mapping = []

        for original_idx, sample in enumerate(self.original_data):
            bb = sample['instructions']

            if not self.enable_chunking or len(bb) <= self.max_instr_count:

                sample_mapping.append({
                    'original_idx': original_idx,
                    'chunk_id': 0,
                    'total_chunks': 1,
                    'is_chunked': False
                })
            else:

                if self.window_size == 0:
                    num_chunks = (len(bb) + self.max_instr_count - 1) // self.max_instr_count
                    for chunk_id in range(num_chunks):
                        sample_mapping.append({
                            'original_idx': original_idx,
                            'chunk_id': chunk_id,
                            'total_chunks': num_chunks,
                            'is_chunked': True
                        })
                else:

                    def _mod(a, b):
                        return a % b or b

                    original_length = len(bb)
                    original_size = 3
                    pad_front = self.window_size - original_size
                    divided = original_length - original_size
                    mod = _mod(divided, self.step_size)
                    pad_back = self.window_size - mod
                    padded_length = pad_front + original_length + pad_back

                    num_windows = (padded_length - self.window_size) // self.step_size + 1

                    for chunk_id in range(num_windows):
                        sample_mapping.append({
                            'original_idx': original_idx,
                            'chunk_id': chunk_id,
                            'total_chunks': num_windows,
                            'is_chunked': True
                        })

        return sample_mapping

    def _process_single_sample(self, original_idx, chunk_id, is_chunked):

        sample = self.original_data[original_idx]
        bb = sample['instructions']
        encoded = [self.tokenizer.encode_instruction(i) for i in bb]

        if not is_chunked:

            processed_x = self._pad_single_sample(encoded)
            chunk_info = {
                'is_chunked': False,
                'chunk_id': 0,
                'total_chunks': 1,
                'original_length': len(encoded),
                'window_size': self.window_size,
                'step_size': self.step_size
            }
        else:

            if self.window_size == 0:

                chunks = self._create_chunks(encoded)
                chunk_data = chunks[chunk_id]
                processed_x = self._pad_single_sample(chunk_data)
                chunk_info = {
                    'is_chunked': True,
                    'chunk_id': chunk_id,
                    'total_chunks': len(chunks),
                    'original_length': len(encoded),
                    'window_size': self.window_size,
                    'step_size': self.step_size
                }
            else:

                chunks = self._create_sliding_windows(encoded)
                chunk_data = chunks[chunk_id]
                processed_x = self._pad_single_sample(chunk_data)
                chunk_info = {
                    'is_chunked': True,
                    'chunk_id': chunk_id,
                    'total_chunks': int(self.window_size / self.step_size),
                    'original_length': len(encoded),
                    'window_size': self.window_size,
                    'step_size': self.step_size
                }

        return {
            'X': processed_x,
            'Y': sample.get('throughput'),
            'idx': sample['idx'],
            'original_idx': original_idx,
            'chunk_info': chunk_info
        }

    def _create_sliding_windows(self, encoded):
        def _mod(a, b):
            return a % b or b

        original_length = len(encoded)
        original_size = 3
        pad_front = self.window_size - original_size
        divided = original_length - original_size
        mod = _mod(divided, self.step_size)
        pad_back = self.window_size - mod

        padding_instr = [self.pad_idx] * self.max_instr_length
        padded_sequence = ([padding_instr] * pad_front +
                           encoded +
                           [padding_instr] * pad_back)

        windows = []
        padded_length = len(padded_sequence)

        for start in range(0, padded_length - self.window_size + 1, self.step_size):
            end = start + self.window_size
            window_sequence = padded_sequence[start:end]
            windows.append(window_sequence)

        return windows

    # def _create_chunks(self, encoded):
    #
    #     chunks = []
    #     start = 0
    #     while start < len(encoded):
    #         end = min(start + self.max_instr_count, len(encoded))
    #         chunks.append(encoded[start:end])
    #         start = end
    #     return chunks

    def _create_chunks(self, encoded):

        total_length = len(encoded)
        num_chunks = (total_length + self.max_instr_count - 1) // self.max_instr_count

        base_size = total_length // num_chunks
        remainder = total_length % num_chunks

        chunks = []
        start = 0

        for i in range(num_chunks):
            chunk_size = base_size + (1 if i < remainder else 0)
            end = start + chunk_size
            chunks.append(encoded[start:end])
            start = end

        return chunks

    def _pad_single_sample(self, encoded):

        if len(encoded) > self.max_instr_count:
            encoded = encoded[:self.max_instr_count]

        padded_encoded = [
            F.pad(torch.tensor(instr, dtype=torch.long), (0, self.max_instr_length - len(instr)),
                  value=self.pad_idx)
            for instr in encoded
        ]

        if len(padded_encoded) < self.max_instr_count:
            padding = torch.full((self.max_instr_count - len(padded_encoded), self.max_instr_length), self.pad_idx,
                                 dtype=torch.long)
            padded_encoded.extend(padding)

        return torch.stack(padded_encoded)

    def __len__(self):
        return len(self.sample_mapping)

    def __getitem__(self, index):

        mapping = self.sample_mapping[index]
        return self._process_single_sample(
            mapping['original_idx'],
            mapping['chunk_id'],
            mapping['is_chunked']
        )

def collate_fn_lstm(batch):
    xs = [item['X'] for item in batch]
    ys = [item['Y'] for item in batch]
    idx = [item['idx'] for item in batch]
    chunk_infos = [item['chunk_info'] for item in batch]

    xs_batch = torch.stack(xs)

    if all(y is None for y in ys):
        ys_batch = ys
    else:
        ys_batch = torch.tensor(ys, dtype=torch.float)

    idx_batch = torch.tensor(idx, dtype=torch.long)

    return {
        'X': xs_batch,
        'Y': ys_batch,
        'idx': idx_batch,
        'chunk_info': chunk_infos
    }