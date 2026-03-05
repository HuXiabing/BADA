import os
import json
import torch
import torch_geometric
from typing import Dict, List
import re
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm

class RISCVGraphEncoder:

    def __init__(self, predefined_token_map="data/vocab.dump"):

        self.node_types = {
            'mnemonic': 0,  # Instruction mnemonic (e.g., 'addi')
            'register': 1,  # Register (e.g., 'x1')
            'immediate': 2,  # Immediate value
            'memory': 3,  # Memory value
            'address': 4,  # Address calculation
            'prefix': 5,  # Instruction prefix
        }

        self.edge_types = {
            'structural': 0,
            'input': 1,
            'output': 2,
            'address_base': 3,
            'address_offset': 4,
        }

        if predefined_token_map is not None:
            self.token_to_idx = torch.load(predefined_token_map)
        else:
            self.token_to_idx = {'<PAD>': 0, '<BLOCK_START>': 1, '<BLOCK_END>': 2, '<ADDRESS>': 3, '<E>': 4, '<D>': 5, '<S>': 6, '<CONST>': 7, '<CSR>': 8,
        'zero': 9, 'ra': 10, 'sp': 11, 'gp': 12, 'tp': 13, 't0': 14, 't1': 15, 't2': 16, 's0': 17, 's1': 18, 'a0': 19, 'a1': 20,
        'a2': 21, 'a3': 22, 'a4': 23, 'a5': 24, 'a6': 25, 'a7': 26, 's2': 27, 's3': 28, 's4': 29, 's5': 30, 's6': 31, 's7': 32,
        's8': 33, 's9': 34, 's10': 35, 's11': 36, 't3': 37, 't4': 38, 't5': 39, 't6': 40, 'ft0': 41, 'ft1': 42, 'ft2': 43, 'ft3': 44,
        'ft4': 45, 'ft5': 46, 'ft6': 47, 'ft7': 48, 'fs0': 49, 'fs1': 50, 'fa0': 51, 'fa1': 52, 'fa2': 53, 'fa3': 54, 'fa4': 55,
        'fa5': 56, 'fa6': 57, 'fa7': 58, 'fs2': 59, 'fs3': 60, 'fs4': 61, 'fs5': 62, 'fs6': 63, 'fs7': 64, 'fs8': 65, 'fs9': 66,
        'fs10': 67, 'fs11': 68, 'ft8': 69, 'ft9': 70, 'ft10': 71, 'ft11': 72, 'x0': 9, 'x1': 10, 'x2': 11, 'x3': 12, 'x4': 13,
        'x5': 14, 'x6': 15, 'x7': 16, 'x8': 17, 'x9': 18, 'x10': 19, 'x11': 20, 'x12': 21, 'x13': 22, 'x14': 23, 'x15': 24, 'x16': 25,
        'x17': 26, 'x18': 27, 'x19': 28, 'x20': 29, 'x21': 30, 'x22': 31, 'x23': 32, 'x24': 33, 'x25': 34, 'x26': 35, 'x27': 36,
        'x28': 37, 'x29': 38, 'x30': 39, 'x31': 40, 'f0': 41, 'f1': 42, 'f2': 43, 'f3': 44, 'f4': 45, 'f5': 46, 'f6': 47, 'f7': 48,
        'f8': 49, 'f9': 50, 'f10': 51, 'f11': 52, 'f12': 53, 'f13': 54, 'f14': 55, 'f15': 56, 'f16': 57, 'f17': 58, 'f18': 59,
        'f19': 60, 'f20': 61, 'f21': 62, 'f22': 63, 'f23': 64, 'f24': 65, 'f25': 66, 'f26': 67, 'f27': 68, 'f28': 69, 'f29': 70,
        'f30': 71, 'f31': 72, 'div': 125, 'divu': 126, 'divuw': 127, 'divw': 128, 'mul': 129, 'mulh': 130, 'mulhsu': 131, 'mulhu': 132,
        'mulw': 133, 'rem': 134, 'remu': 135, 'remuw': 136, 'remw': 137, 'add': 73, 'addi': 74, 'addiw': 75, 'addw': 76, 'and': 77,
        'andi': 78, 'auipc': 79, 'beq': 80, 'bge': 81, 'bgeu': 82, 'blt': 83, 'bltu': 84, 'bne': 85, 'ebreak': 86, 'ecall': 87,
        'fence': 88, 'jal': 89, 'jalr': 90, 'lb': 91, 'lbu': 92, 'ld': 93, 'lh': 94, 'lhu': 95, 'lui': 96, 'lw': 97, 'lwu': 98,
        'or': 99, 'ori': 100, 'sb': 101, 'sd': 102, 'sh': 103, 'sll': 104, 'slli': 105, 'slliw': 106, 'sllw': 107, 'slt': 108,
        'slti': 109, 'sltiu': 110, 'sltu': 111, 'sra': 112, 'srai': 113, 'sraiw': 114, 'sraw': 115, 'srl': 116, 'srli': 117,
        'srliw': 118, 'srlw': 119, 'sub': 120, 'subw': 121, 'sw': 122, 'xor': 123, 'xori': 124}

        self.mnemonic_to_token = {'div': 125, 'divu': 126, 'divuw': 127, 'divw': 128, 'mul': 129, 'mulh': 130, 'mulhsu': 131, 'mulhu': 132,
        'mulw': 133, 'rem': 134, 'remu': 135, 'remuw': 136, 'remw': 137, 'add': 73, 'addi': 74, 'addiw': 75, 'addw': 76, 'and': 77,
        'andi': 78, 'auipc': 79, 'beq': 80, 'bge': 81, 'bgeu': 82, 'blt': 83, 'bltu': 84, 'bne': 85, 'ebreak': 86, 'ecall': 87,
        'fence': 88, 'jal': 89, 'jalr': 90, 'lb': 91, 'lbu': 92, 'ld': 93, 'lh': 94, 'lhu': 95, 'lui': 96, 'lw': 97, 'lwu': 98,
        'or': 99, 'ori': 100, 'sb': 101, 'sd': 102, 'sh': 103, 'sll': 104, 'slli': 105, 'slliw': 106, 'sllw': 107, 'slt': 108,
        'slti': 109, 'sltiu': 110, 'sltu': 111, 'sra': 112, 'srai': 113, 'sraiw': 114, 'sraw': 115, 'srl': 116, 'srli': 117,
        'srliw': 118, 'srlw': 119, 'sub': 120, 'subw': 121, 'sw': 122, 'xor': 123, 'xori': 124}

    def parse_instruction(self, instruction: str) -> Dict:

        instruction = instruction.strip().lower()
        match = re.match(r'([a-z0-9\.]+)\s*(.*)', instruction)
        if not match:
            return {'mnemonic': '<UNK>', 'operands': []}

        mnemonic, operands_str = match.groups()
        operands = [op.strip() for op in operands_str.split(',')] if operands_str else []

        return {
            'mnemonic': mnemonic,
            'operands': operands
        }

    def get_token_id(self, token: str, token_type: str = None) -> int:

        token = token.lower()

        if token_type == 'mnemonic':
            return self.mnemonic_to_token.get(token, 0)

        return self.token_to_idx.get(token, 0)

    def get_vocab_size(self):
        return len(self.token_to_idx)

    def get_num_edge_types(self):
        return len(self.edge_types)

    def build_graph(self, basic_block: List[str], encoded_tokens: List[List[int]] = None) -> torch_geometric.data.Data:

        nodes = []  # (token_id)
        edges = []  # (src, dst, type)
        instruction_token_ids = []

        value_nodes = {}
        instruction_nodes = []
        node_idx = 0

        for instr_idx, instruction in enumerate(basic_block):
            parsed = self.parse_instruction(instruction)
            mnemonic = parsed['mnemonic']
            operands = parsed['operands']

            if encoded_tokens is not None and instr_idx < len(encoded_tokens):
                token_id = encoded_tokens[instr_idx][0] if encoded_tokens[instr_idx] else 0
            else:
                token_id = self.get_token_id(mnemonic, 'mnemonic')

            instruction_token_ids.append(token_id)
            mnemonic_node_idx = node_idx
            nodes.append(token_id)
            instruction_nodes.append(mnemonic_node_idx)
            node_idx += 1

            if instr_idx > 0:
                edges.append((instruction_nodes[instr_idx - 1], mnemonic_node_idx, self.edge_types['structural']))

            if operands:
                dest_operand = operands[0]

                if dest_operand in self.token_to_idx and (
                        dest_operand.startswith(('x', 'a', 's', 't')) or
                        dest_operand in ('ra', 'sp', 'gp', 'tp', 'fp', 'zero')):
                    dest_node_idx = node_idx
                    nodes.append(self.token_to_idx.get(dest_operand, 0))
                    node_idx += 1

                    edges.append((mnemonic_node_idx, dest_node_idx, self.edge_types['output']))
                    value_nodes[dest_operand] = dest_node_idx

                elif '(' in dest_operand and ')' in dest_operand: # "sw x1, 8(x2)"
                    match = re.match(r'(\d+)\(([^\)]+)\)', dest_operand)
                    if match:
                        offset, base_reg = match.groups()

                        addr_node_idx = node_idx
                        nodes.append(self.token_to_idx.get('<ADDRESS>', 0))
                        node_idx += 1

                        mem_node_idx = node_idx
                        nodes.append(self.token_to_idx.get('<CONST>', 0))
                        node_idx += 1

                        if base_reg in value_nodes:
                            edges.append((value_nodes[base_reg], addr_node_idx, self.edge_types['address_base']))

                        imm_node_idx = node_idx
                        nodes.append(self.token_to_idx.get('<CONST>', 0))
                        node_idx += 1
                        edges.append((imm_node_idx, addr_node_idx, self.edge_types['address_offset']))
                        edges.append((mnemonic_node_idx, mem_node_idx, self.edge_types['output']))

            for src_idx, src_operand in enumerate(operands[1:], 1):
                if src_operand in self.token_to_idx and (
                        src_operand.startswith(('x', 'a', 's', 't')) or
                        src_operand in ('ra', 'sp', 'gp', 'tp', 'fp', 'zero')):
                    if src_operand in value_nodes:
                        src_node_idx = value_nodes[src_operand]
                    else:
                        src_node_idx = node_idx
                        nodes.append(self.token_to_idx.get(src_operand, 0))
                        value_nodes[src_operand] = src_node_idx
                        node_idx += 1

                    edges.append((src_node_idx, mnemonic_node_idx, self.edge_types['input']))

                elif src_operand.lstrip('-').isdigit() or (
                        src_operand.startswith('0x') and all(c in '0123456789abcdefABCDEF' for c in src_operand[2:])):
                    imm_node_idx = node_idx
                    nodes.append(self.token_to_idx.get('<CONST>', 0))
                    node_idx += 1
                    edges.append((imm_node_idx, mnemonic_node_idx, self.edge_types['input']))

                elif '(' in src_operand and ')' in src_operand:
                    match = re.match(r'(\d+)\(([^\)]+)\)', src_operand)
                    if match:
                        offset, base_reg = match.groups()

                        addr_node_idx = node_idx
                        nodes.append(self.token_to_idx.get('<ADDRESS>', 0))
                        node_idx += 1

                        mem_node_idx = node_idx
                        nodes.append(self.token_to_idx.get('<CONST>', 0))
                        node_idx += 1

                        if base_reg in value_nodes:
                            edges.append((value_nodes[base_reg], addr_node_idx, self.edge_types['address_base']))

                        imm_node_idx = node_idx
                        nodes.append(self.token_to_idx.get('<CONST>', 0))
                        node_idx += 1
                        edges.append((imm_node_idx, addr_node_idx, self.edge_types['address_offset']))
                        edges.append((mem_node_idx, mnemonic_node_idx, self.edge_types['input']))

        x = torch.tensor(nodes, dtype=torch.long)  # [num_nodes]
        edge_index = torch.zeros((2, len(edges)), dtype=torch.long)
        edge_attr = torch.zeros((len(edges), 1), dtype=torch.long)

        for i, (src, dst, edge_type) in enumerate(edges):
            edge_index[0, i] = src
            edge_index[1, i] = dst
            edge_attr[i, 0] = edge_type

        instruction_mask = torch.zeros(len(nodes), dtype=torch.bool)
        for idx in instruction_nodes:
            instruction_mask[idx] = True

        instruction_token_ids_tensor = torch.tensor(instruction_token_ids, dtype=torch.long)

        data = torch_geometric.data.Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            instruction_mask=instruction_mask,
            instruction_token_ids=instruction_token_ids_tensor,
            num_nodes=len(nodes)
        )

        return data

class RISCVGraphDataset(Dataset):

    def __init__(self, json_path,
                 cache_dir=None,
                 rebuild_cache=False,
                 max_instr_count=64,
                 enable_chunking=False,
                 window_size=0,
                 step_size=0):

        self.json_path = json_path
        self.cache_dir = cache_dir
        self.max_instr_count = max_instr_count
        self.enable_chunking = enable_chunking
        self.window_size = window_size
        self.step_size = step_size
        self.graph_encoder = RISCVGraphEncoder()

        cache_file = None
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            json_filename = os.path.basename(json_path)
            cache_suffix = f"_chunk{max_instr_count}_win{window_size}_step{step_size}" if enable_chunking else ""
            cache_file = os.path.join(cache_dir,
                                      f"{os.path.splitext(json_filename)[0]}_sample_mapping{cache_suffix}.pkl")

        if cache_file is not None and os.path.exists(cache_file) and not rebuild_cache:
            print(f"Loading sample mapping from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.sample_mapping = cached_data['sample_mapping']
                self.original_data = cached_data['original_data']
                print(f"Loaded {len(self.sample_mapping)} samples from cache")
        else:
            print(f"Loading data from: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as file:
                self.original_data = json.load(file)

            print("Creating sample mapping...")
            self.sample_mapping = self._create_sample_mapping()

            if cache_file is not None:
                print(f"Saving sample mapping to cache: {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'sample_mapping': self.sample_mapping,
                        'original_data': self.original_data,
                    }, f)
                print(f"Saved {len(self.sample_mapping)} samples to cache")

        print(f"Dataset initialized with {len(self.sample_mapping)} samples")

    def _create_sample_mapping(self):

        sample_mapping = []
        for original_idx, sample in enumerate(tqdm(self.original_data, desc="Creating sample mapping")):
            instructions = sample.get('instructions', [])

            if not instructions:
                continue

            if not self.enable_chunking or len(instructions) <= self.max_instr_count:
                sample_mapping.append({
                    'original_idx': original_idx,
                    'chunk_id': 0,
                    'total_chunks': 1,
                    'is_chunked': False,
                    'instructions': instructions,
                    'original_length': len(instructions)
                })
            else:
                if self.window_size == 0:
                    chunks = self._create_chunks(instructions)
                    for chunk_id, chunk in enumerate(chunks):
                        if chunk:
                            sample_mapping.append({
                                'original_idx': original_idx,
                                'chunk_id': chunk_id,
                                'total_chunks': len(chunks),
                                'is_chunked': True,
                                'instructions': chunk,
                                'original_length': len(instructions)
                            })
                else:
                    chunks = self._create_sliding_windows(instructions)
                    for chunk_id, chunk in enumerate(chunks):
                        if chunk:
                            sample_mapping.append({
                                'original_idx': original_idx,
                                'chunk_id': chunk_id,
                                'total_chunks': len(chunks),
                                'is_chunked': True,
                                'instructions': chunk,
                                'original_length': len(instructions)
                            })
        return sample_mapping

    def _build_graph_for_sample(self, instructions):

        graph = self.graph_encoder.build_graph(instructions)
        instruction_mask = graph.instruction_mask
        instruction_nodes = torch.where(instruction_mask)[0]
        instruction_token_ids = graph.x[instruction_nodes]
        graph.instruction_token_ids = instruction_token_ids

        return graph

    def _create_chunks(self, instructions):

        chunks = []
        start = 0
        while start < len(instructions):
            end = min(start + self.max_instr_count, len(instructions))
            chunks.append(instructions[start:end])
            start = end
        return chunks

    def _create_sliding_windows(self, instructions):

        def _mod(a, b):
            return a % b or b

        original_length = len(instructions)
        original_size = 3
        pad_front = self.window_size - original_size
        divided = original_length - original_size
        mod = _mod(divided, self.step_size)
        pad_back = self.window_size - mod

        padding_instr = ""
        padded_sequence = ([padding_instr] * pad_front +
                           instructions +
                           [padding_instr] * pad_back)

        windows = []
        padded_length = len(padded_sequence)

        for start in range(0, padded_length - self.window_size + 1, self.step_size):
            end = start + self.window_size
            window_sequence = padded_sequence[start:end]
            window_sequence = [instr for instr in window_sequence if instr.strip()]
            if window_sequence:
                windows.append(window_sequence)

        return windows

    def __len__(self):
        return len(self.sample_mapping)

    def __getitem__(self, idx):
        mapping_info = self.sample_mapping[idx]
        original_sample = self.original_data[mapping_info['original_idx']]
        graph = self._build_graph_for_sample(mapping_info['instructions'])

        if mapping_info['is_chunked'] and self.window_size:
            mapping_info['total_chunks'] = int(self.window_size / self.step_size)
            # print(f"sliding window size: {mapping_info['total_chunks']}")
        chunk_info = {
            'is_chunked': mapping_info['is_chunked'],
            'chunk_id': mapping_info['chunk_id'],
            'total_chunks': mapping_info['total_chunks'],
            'original_length': mapping_info['original_length'],
            'window_size': self.window_size,
            'step_size': self.step_size,
            'original_idx': mapping_info['original_idx']
        }

        return {
            'X': graph,
            'idx': original_sample['idx'],
            'Y': original_sample.get('throughput'),
            'chunk_info': json.dumps(chunk_info),
            'raw_instructions': mapping_info['instructions']
        }

    def get_dataset_info(self):
        total_samples = len(self.sample_mapping)
        chunked_samples = sum(1 for mapping in self.sample_mapping if mapping['is_chunked'])
        original_samples = len(self.original_data)

        return {
            'total_samples': total_samples,
            'original_samples': original_samples,
            'chunked_samples': chunked_samples,
            'chunking_enabled': self.enable_chunking,
            'max_instr_count': self.max_instr_count,
            'window_size': self.window_size,
            'step_size': self.step_size
        }

if __name__ == '__main__':
    encoder = RISCVGraphEncoder()
    instruction = ['addi	sp,sp,-32','sd	s0,16(sp)','ld	s2,8(a4)','auipc	ra,0x390']
    # for i in instruction:
    #     parsed = encoder.parse_instruction(i)
    #     print(parsed)
    #     token_id = encoder.get_token_id(parsed['mnemonic'], 'mnemonic')
    #     print(token_id)
    graph = encoder.build_graph(instruction) # torch_geometric.data.Data
    print(graph.instruction_token_ids)
