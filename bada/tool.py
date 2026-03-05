import argparse
import numpy as np
import pandas as pd
import json
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import re
import time
import hashlib
from typing import List, Dict, Tuple, Any, Set, Optional
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
from collections import defaultdict
from bada.analyzer import DataDependencyAnalyzer
from bada.generator import OptimizedPreciseBasicBlockGenerator
from scipy.stats import norm, qmc
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Serif CJK JP']
plt.rcParams['axes.unicode_minus'] = False
import subprocess
import tempfile
import os

class RISCVGenerator:
    def __init__(self):
        self.instruction_classes = {
            'shifts': {
                'shift_r': ['sll', 'sllw', 'sra', 'sraw', 'srl', 'srlw'],
                'shift_i': ['slli', 'slliw', 'srai', 'sraiw', 'srli', 'srliw']
            },
            'arithmetic': {
                'arith_r': ['add', 'addw', 'sub', 'subw'],
                'arith_i': ['addi', 'addiw'],
                'arith_u': ['auipc', 'lui']
            },
            'logical': {
                'logic_r': ['and', 'xor', 'or'],
                'logic_i': ['andi', 'ori', 'xori']
            },
            'compare': {
                'compare_r': ['slt', 'sltu'],
                'compare_i': ['slti', 'sltiu']
            },
            'mul': {
                'multiply': ['mul', 'mulh', 'mulhsu', 'mulhu', 'mulw']
            },
            'div': {
                'divide': ['div', 'divu', 'divuw', 'divw']
            },
            'rem': {
                'remainder': ['rem', 'remu', 'remuw', 'remw']
            },
            'load': {
                'load_mem': ['lb', 'lbu', 'ld', 'lh', 'lhu', 'lw', 'lwu']
            },
            'store': {
                'store_mem': ['sb', 'sd', 'sh', 'sw']
            }
        }

        self.all_instruction_types = []
        for main_class, subclasses in self.instruction_classes.items():
            for subclass_name, instruction_list in subclasses.items():
                self.all_instruction_types.extend(instruction_list)

        self.inst_to_index = {inst: i for i, inst in enumerate(self.all_instruction_types)}

        self.inst_to_subclass = {}
        self.subclass_list = []
        for main_class, subclasses in self.instruction_classes.items():
            for subclass_name, instructions in subclasses.items():
                self.subclass_list.append(subclass_name)
                for inst in instructions:
                    self.inst_to_subclass[inst] = subclass_name

        self.feature_names = self.all_instruction_types + ['waw_deps', 'raw_deps', 'war_deps']

        self.registers = {
            'a': [f'a{i}' for i in range(8)],
            't': [f't{i}' for i in range(7)],
            's': [f's{i}' for i in range(1, 12)],
            'special': ['gp', 'sp', 'ra', 'tp', 'zero', 'fp']
        }
        self.all_registers = []
        for reg_type, reg_list in self.registers.items():
            self.all_registers.extend(reg_list)
        self.writable_registers = [reg for reg in self.all_registers if reg != 'zero']
        self.precise_generator = OptimizedPreciseBasicBlockGenerator()
        self.dependency_analyzer = DataDependencyAnalyzer()
        self.gp_model = None
        self.scaler = StandardScaler()
        self.generated_hashes = set()
        self.duplicate_count = 0
        self.timing_stats = {}

    def sample_to_feature_vector(self, sample: Dict) -> np.ndarray:
        """
        Convert sample to 57-dimensional feature vector:
        - First 54 dimensions: count of each instruction type
        - Last 3 dimensions: normalized dependency ratios (waw, raw, war)
        """
        instructions = sample['instructions']
        feature_vector = np.zeros(57)  # 54 instruction types + 3 dependencies

        for instruction in instructions:
            inst_type = self.extract_instruction_type(instruction)
            if inst_type in self.inst_to_index:
                feature_vector[self.inst_to_index[inst_type]] += 1

        length = sum(feature_vector[:54])

        if length == 0:
            return np.zeros(57)

        dependencies = self.dependency_analyzer.analyze_dependencies(instructions)
        feature_vector[54] = dependencies['waw'] / length  # waw_deps
        feature_vector[55] = dependencies['raw'] / length  # raw_deps
        feature_vector[56] = dependencies['war'] / length  # war_deps

        if feature_vector[54] > 100 or feature_vector[55] > 100 or feature_vector[56] > 100 or length > 128:
            # print(instructions)
            return np.zeros(57)

        return feature_vector

    def extract_instruction_type(self, instruction: str) -> str:

        instruction = instruction.strip()
        parts = instruction.replace('\t', ' ').split()
        if parts:
            return parts[0].lower()
        return "unknown"

    def feature_to_multiple_basic_blocks(self, feature_vector: np.ndarray,
                                         n_variants: int = 20,
                                         base_seed: int = None,
                                         verbose: bool = False) -> List[List[str]]:

        if base_seed is not None:
            np.random.seed(base_seed)

        variants = []
        variant_hashes = set()

        for variant_id in range(n_variants * 3):
            if len(variants) >= n_variants:
                break

            base_instructions = self.precise_generator.generate_precise_basic_block(feature_vector, verbose=0)
            depth = self.dependency_analyzer.analyze_dependencies(base_instructions)
            if verbose:
                print(
                    f"Object:, WAW={int(feature_vector[-3])}, RAW={int(feature_vector[-2])}, WAR={int(feature_vector[-1])}")
                print(f"generating waw={depth['waw']}, raw={depth['raw']}, war={depth['war']}")

            final_instructions = self._apply_diversity_transforms(base_instructions)

            depth = self.dependency_analyzer.analyze_dependencies(final_instructions)
            if verbose:
                print(f"fine tuning waw={depth['waw']}, raw={depth['raw']}, war={depth['war']}")

            variant_hash = self.compute_sample_hash(final_instructions)
            if variant_hash not in variant_hashes:
                variants.append(final_instructions)
                variant_hashes.add(variant_hash)

        return variants[:n_variants]

    def _apply_diversity_transforms(self, instructions: List[str]) -> List[str]:
        if np.random.random() < 0.3:
            transform_type = np.random.choice([
                'register_aliasing', 'instruction_variation'
            ])

            if transform_type == 'register_aliasing':
                return self._apply_register_aliasing(instructions)
            else:  # instruction_variation
                return self._apply_instruction_variation(instructions)
        else:
            return instructions

    def _apply_register_aliasing(self, instructions: List[str]) -> List[str]:

        modified_instructions = []

        interchangeable_groups = [
            ['t0', 't1', 't2'],
            ['a0', 'a1', 'a2'],
            ['s1', 's2', 's3'],
        ]

        if np.random.random() < 0.7:
            target_group = np.random.choice(len(interchangeable_groups))
            reg_group = interchangeable_groups[target_group]

            shuffled_group = reg_group[:]
            np.random.shuffle(shuffled_group)
            alias_map = dict(zip(reg_group, shuffled_group))

            for inst in instructions:
                modified_inst = inst
                for old_reg, new_reg in alias_map.items():
                    modified_inst = re.sub(r'\b' + re.escape(old_reg) + r'\b', new_reg, modified_inst)
                modified_instructions.append(modified_inst)
        else:
            modified_instructions = instructions[:]

        return modified_instructions

    def _apply_instruction_variation(self, instructions: List[str]) -> List[str]:

        modified_instructions = []
        similar_instructions = {
            'add': ['add', 'sub'],
            'sub': ['sub', 'add'],
            'and': ['and', 'or', 'xor'],
            'or': ['or', 'and', 'xor'],
            'xor': ['xor', 'and', 'or'],
            'sll': ['sll', 'srl'],
            'srl': ['srl', 'sll'],
        }

        for inst in instructions:
            modified_inst = inst
            inst_name = inst.split()[0].lower() if inst.split() else ""

            if inst_name in similar_instructions and np.random.random() < 0.2:
                replacement_options = similar_instructions[inst_name]
                new_inst_name = np.random.choice(replacement_options)
                parts = inst.split('\t', 1)
                if len(parts) == 2:
                    modified_inst = f"{new_inst_name}\t{parts[1]}"

            modified_instructions.append(modified_inst)

        return modified_instructions

    def compute_sample_hash(self, instructions: List[str]) -> str:

        content = "\\n".join(sorted(instructions))
        return hashlib.md5(content.encode()).hexdigest()

    def generate_diverse_candidates(self, n_samples: int = 20000,
                                    length_avg_loss: dict = None, seed=71) -> np.ndarray:

        np.random.seed(seed)
        vec_len = 54
        losses = np.array(list(length_avg_loss.values()))
        lengths = np.array(list(length_avg_loss.keys()))
        loss_weights = losses / losses.sum()

        samples_per_length = (loss_weights * n_samples).astype(int)
        diff = n_samples - samples_per_length.sum()
        if diff != 0:
            highest_loss_idx = np.argmax(losses)
            samples_per_length[highest_loss_idx] += diff

        sampler = qmc.LatinHypercube(d=vec_len + 3)
        unit_samples = sampler.random(n_samples)

        all_samples = []
        sample_start_idx = 0

        for i, (length, num_samples) in enumerate(zip(lengths, samples_per_length)):
            if num_samples <= 0:
                continue

            sample_end_idx = sample_start_idx + num_samples
            current_unit_samples = unit_samples[sample_start_idx:sample_end_idx]

            samples = np.zeros((num_samples, vec_len + 3))
            probabilities = current_unit_samples[:, :vec_len]
            probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

            for j in range(num_samples):
                for _ in range(length):
                    chosen_instruction = np.random.choice(vec_len, p=probabilities[j, :])
                    samples[j, chosen_instruction] += 1
            # coefficients = [1.02, 0.80, 0.90]  # 90%
            coefficients = [1.57, 0.89, 1.35]  # 95%
            # coefficients = [3.23, 1.08, 3.03]   # 99%
            for j in range(3):
                samples[:, vec_len + j] = current_unit_samples[:, vec_len + j] * coefficients[j]

            all_samples.append(samples)
            sample_start_idx = sample_end_idx

        if all_samples:
            final_samples = np.vstack(all_samples)
            return final_samples
        else:
            return np.empty((0, vec_len + 3))

    def generate_enhanced_dataset(self, matched_samples: List[Dict],
                                  n_candidates: int = 20000,
                                  n_selected_features: int = 100,
                                  variants_per_feature: int = 20,
                                  max_training_samples: int = 8000,
                                  top_dict: dict = None) -> List[Dict]:

        total_start_time = time.time()

        print("\nData preparation ...")
        X, y, idx = self.prepare_training_data(matched_samples, max_training_samples)
        print("\nTraining model...")
        self.train_optimized_model(X, y)

        print(f"\nGenerating {n_candidates} candidates...")
        candidates = self.generate_diverse_candidates(n_candidates, top_dict)
        candidates = np.vstack((candidates, X))
        unique_candidates = np.unique(candidates, axis=0)
        # Evaluation
        acquisition_values, predicted_means, predicted_stds = self.ucb_acquisition_function(candidates)  #UCB
        # acquisition_values, predicted_means, predicted_stds = self.pi_acquisition_function(candidates, f_best=float(
        #     np.max(y)))  # PI
        # acquisition_values, predicted_means, predicted_stds = self.ei_acquisition_function(
        #     unique_candidates, f_best = float(np.max(y)), batch_size = 10000)  # EI

        top_indices = np.argsort(acquisition_values)[-n_selected_features:][::-1]
        selected_features = candidates[top_indices]

        print(f"\n Generating basic blocks...")
        augmented_dataset = []

        for feature_idx, feature_vector in enumerate(selected_features):
            variants = self.feature_to_multiple_basic_blocks(
                feature_vector,
                n_variants=variants_per_feature,
                base_seed=feature_idx * 1000
            )

            for variant_idx, instructions in enumerate(variants):
                actual_deps = self.dependency_analyzer.analyze_dependencies(instructions)
                sample = {
                    # 'feature_vector': feature_vector.tolist(),
                    # 'target_dependencies': {
                    #     'waw': int(feature_vector[-3]),
                    #     'raw': int(feature_vector[-2]),
                    #     'war': int(feature_vector[-1])
                    # },
                    # 'actual_dependencies': actual_deps,
                    # 'dependency_accuracy': self._calculate_dependency_accuracy(
                    #     feature_vector[-3:], actual_deps
                    # ),
                    'feature_id': feature_idx,
                    'variant_id': variant_idx,
                    'asm': "\\n".join(instructions),
                }

                augmented_dataset.append(sample)

        total_time = time.time() - total_start_time

        print(f"Final number of generated samples: {len(augmented_dataset):,}")
        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Average time per sample: {total_time / len(augmented_dataset):.3f} seconds")

        return augmented_dataset

    def _calculate_dependency_accuracy(self, target_deps: np.ndarray,
                                       actual_deps: Dict[str, int]) -> float:

        target = [int(target_deps[0]), int(target_deps[1]), int(target_deps[2])]
        actual = [actual_deps['waw'], actual_deps['raw'], actual_deps['war']]

        total_error = 0
        for t, a in zip(target, actual):
            if t == 0 and a == 0:
                continue
            elif t == 0:
                total_error += 1.0
            else:
                total_error += abs(t - a) / max(t, 1)

        accuracy = max(0, 1 - total_error / 3)
        return accuracy

    def prepare_training_data(self, matched_samples: List[Dict],
                              verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:

        sampled_samples = matched_samples

        X = []
        y = []
        idx = []
        cnt = 0

        for sample in sampled_samples:
            feature_vector = self.sample_to_feature_vector(sample)
            if (feature_vector == np.zeros(57)).all():
                continue
            X.append(feature_vector)
            y.append(sample['loss'])
            idx.append(sample['idx'])

        X = np.array(X)
        y = np.array(y)

        return X, y, idx

    def merge_duplicate_samples(self, X, y, tolerance=1e-10, verbose=1) -> Tuple[np.ndarray, np.ndarray, Dict]:

        start_time = time.time()
        original_samples = X.shape[0]

        if tolerance > 0:
            decimal_places = int(-np.log10(tolerance))
            X_rounded = np.round(X, decimal_places)
        else:
            X_rounded = X.copy()

        sample_groups = defaultdict(list)

        for i, row in enumerate(X_rounded):
            key = tuple(row)
            sample_groups[key].append(y[i])

        X_merged_list = []
        y_merged_list = []

        duplicate_groups = 0
        group_sizes = []
        y_stds = []

        for x_tuple, y_values in sample_groups.items():
            X_merged_list.append(list(x_tuple))
            y_mean = np.median(y_values)
            y_merged_list.append(y_mean)

            if len(y_values) > 1:
                duplicate_groups += 1
                y_stds.append(np.std(y_values))

            group_sizes.append(len(y_values))

        X_merged = np.array(X_merged_list)
        y_merged = np.array(y_merged_list)

        merge_info = {
            'duplicate_groups': duplicate_groups,
            'avg_group_size': np.mean(group_sizes),
            'max_group_size': np.max(group_sizes),
            'avg_y_std': np.mean(y_stds) if y_stds else 0,
            'max_y_std': np.max(y_stds) if y_stds else 0,
        }

        processing_time = time.time() - start_time

        return X_merged, y_merged, merge_info

    def train_optimized_model(self, X: np.ndarray, y: np.ndarray):

        X, y, merge_info = self.merge_duplicate_samples(X, y, verbose=0)

        start_time = time.time()

        X_scaled = self.scaler.fit_transform(X)

        print("Using rf regressor")
        self.gp_model = RandomForestRegressor()
        self.gp_model.fit(X_scaled, y)
        original_model = self.gp_model
        scaler = self.scaler

        class RFWrapper:
            def __init__(self, rf_model, scaler):
                self.rf_model = rf_model
                self.scaler = scaler

            def predict(self, X, return_std=False):
                X_scaled = self.scaler.transform(X)
                y_pred = self.rf_model.predict(X_scaled)

                if return_std:
                    predictions = np.array([
                        tree.predict(X_scaled) for tree in self.rf_model.estimators_
                    ])
                    y_std = np.std(predictions, axis=0)

                    return y_pred, y_std
                else:
                    return y_pred

        self.gp_model = RFWrapper(original_model, scaler)

        y_pred, y_std = self.gp_model.predict(X_scaled, return_std=True)
        mse = np.mean((y - y_pred) ** 2)

        training_time = time.time() - start_time
        self.timing_stats['training'] = training_time
        print(f"rf finished in {training_time:.2f}s！")

        return self.gp_model

    def ucb_acquisition_function(self, X_candidates: np.ndarray,
                                 exploration_weight: float = 2.8,
                                 exploitation_weight: float = 1.2,
                                 batch_size: int = 10000,
                                 verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        start_time = time.time()

        n_candidates = len(X_candidates)
        all_acquisition = []
        all_means = []
        all_stds = []

        for i in range(0, n_candidates, batch_size):
            end_idx = min(i + batch_size, n_candidates)
            batch_candidates = X_candidates[i:end_idx]

            batch_scaled = self.scaler.transform(batch_candidates)
            batch_means, batch_stds = self.gp_model.predict(batch_scaled, return_std=True)

            batch_acquisition = exploitation_weight * batch_means + exploration_weight * batch_stds  # UCB  = mean + kappa * std

            all_acquisition.extend(batch_acquisition)
            all_means.extend(batch_means)
            all_stds.extend(batch_stds)

        acquisition_time = time.time() - start_time
        self.timing_stats['acquisition'] = acquisition_time

        return np.array(all_acquisition), np.array(all_means), np.array(all_stds)

    def ei_acquisition_function(self, X_candidates: np.ndarray,
                                f_best: float,
                                batch_size: int = 10000,
                                verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        n_candidates = len(X_candidates)
        all_acquisition = []
        all_means = []
        all_stds = []

        min_std = 1e-9

        for i in range(0, n_candidates, batch_size):
            end_idx = min(i + batch_size, n_candidates)
            batch_candidates = X_candidates[i:end_idx]
            batch_scaled = self.scaler.transform(batch_candidates)
            batch_means, batch_stds = self.gp_model.predict(batch_scaled, return_std=True)

            batch_acquisition = self._compute_ei(batch_means, batch_stds, f_best, min_std)

            all_acquisition.extend(batch_acquisition)
            all_means.extend(batch_means)
            all_stds.extend(batch_stds)

            if verbose and i % (batch_size * 10) == 0:
                print(f"Processed {min(end_idx, n_candidates)}/{n_candidates} candidates")

        return np.array(all_acquisition), np.array(all_means), np.array(all_stds)

    def _compute_ei(self, means: np.ndarray, stds: np.ndarray,
                    f_best: float, min_std: float = 1e-9) -> np.ndarray:
        """
        Expected Improvement
        """
        acquisition = np.zeros_like(means)

        valid_mask = stds > min_std

        if not np.any(valid_mask):
            return np.maximum(means - f_best, 0)

        valid_means = means[valid_mask]
        valid_stds = stds[valid_mask]

        improvement = valid_means - f_best
        Z = improvement / valid_stds

        ei_values = improvement * norm.cdf(Z) + valid_stds * norm.pdf(Z)
        ei_values = np.maximum(ei_values, 0)

        acquisition[valid_mask] = ei_values

        return acquisition

    def pi_acquisition_function(self, X_candidates: np.ndarray,
                                f_best: float,
                                xi: float = 0.01,
                                batch_size: int = 10000,
                                verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        n_candidates = len(X_candidates)
        all_acquisition = []
        all_means = []
        all_stds = []

        min_std = 1e-9  # Minimum standard deviation for numerical stability

        for i in range(0, n_candidates, batch_size):
            end_idx = min(i + batch_size, n_candidates)
            batch_candidates = X_candidates[i:end_idx]
            batch_scaled = self.scaler.transform(batch_candidates)
            batch_means, batch_stds = self.gp_model.predict(batch_scaled, return_std=True)

            batch_acquisition = self._compute_pi(batch_means, batch_stds, f_best, xi, min_std)

            all_acquisition.extend(batch_acquisition)
            all_means.extend(batch_means)
            all_stds.extend(batch_stds)

            if verbose and i % (batch_size * 10) == 0 and i > 0:
                print(f"Processed {min(end_idx, n_candidates)}/{n_candidates} candidates")

        return np.array(all_acquisition), np.array(all_means), np.array(all_stds)

    def _compute_pi(self, means: np.ndarray, stds: np.ndarray,
                    f_best: float, xi: float = 0.01,
                    min_std: float = 1e-9) -> np.ndarray:
        """
        Probability of Improvement for maximization problems

        PI = P(f(x) >= f_best + xi)
        """
        pi_values = np.zeros_like(means)

        valid_mask = stds > min_std

        if not np.any(valid_mask):
            improvement = means - f_best
            return (improvement > 0).astype(float)

        mu = means[valid_mask]
        sigma = stds[valid_mask]

        Z = (mu - f_best - xi) / sigma
        pi_values[valid_mask] = norm.cdf(Z)

        # Points with very small std
        small_std_mask = ~valid_mask
        if np.any(small_std_mask):
            improvement = means[small_std_mask] - f_best
            pi_values[small_std_mask] = (improvement > 0).astype(float)

        return pi_values

    def save_generated_data(self, augmented_dataset: List[Dict], exp_name: str):
        with open(f"../experiments/{exp_name}/input_generated.json", 'w', encoding='utf-8') as f:
            json.dump(augmented_dataset, f, ensure_ascii=False, indent=2)

    def save_generated_data_hardware(self, augmented_dataset: List[Dict], exp_name: str):

        raw_data = []
        for i, sample in enumerate(augmented_dataset):
            asm = sample['asm']
            machine_code = asm_to_bin(asm)
            if not machine_code:
                continue

            data_entry = {
                "asm": asm,
                "binary": machine_code,
                "idx": i
            }
            raw_data.append(data_entry)

        with open(f"../experiments/{exp_name}/input_generated.json", 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)


def workflow(exp_name: str, csv_file_path: str, json_file_path: str,
             n_candidates: int = 20000,
             n_selected_features: int = 100,
             variants_per_feature: int = 20,
             max_training_samples: int = 8000,
             hardware: bool = True):
    matched_samples, top_dict = match_samples(csv_file_path, json_file_path, sample_method='top')

    if not matched_samples:
        print("Error! Didn't find any samples matching")
        return []

    generator = RISCVGenerator()

    augmented_dataset = generator.generate_enhanced_dataset(
        matched_samples=matched_samples,
        n_candidates=n_candidates,
        n_selected_features=n_selected_features,
        variants_per_feature=variants_per_feature,
        max_training_samples=max_training_samples,
        top_dict=top_dict
    )

    if hardware:
        generator.save_generated_data_hardware(augmented_dataset, exp_name)
    else:
        generator.save_generated_data(augmented_dataset, exp_name)
    return augmented_dataset


def asm_to_bin(instructions_str):
    instructions = instructions_str.replace('\\n', '\n').replace('\\t', '\t')

    with tempfile.NamedTemporaryFile(mode='w', suffix='.s', delete=False) as temp_s:
        temp_s_name = temp_s.name
        temp_s.write('.section .text\n')
        temp_s.write('.globl _start\n')
        temp_s.write('_start:\n')
        temp_s.write(instructions)
        temp_s.write('\n')

    temp_o_name = temp_s_name.replace('.s', '.o')

    try:
        result_asm = subprocess.run(
            ['riscv64-unknown-linux-gnu-as', '-march=rv64g', '-o', temp_o_name, temp_s_name],
            capture_output=True,
            text=True
        )

        if result_asm.returncode != 0:
            print("=== Assembly file content ===")
            with open(temp_s_name, 'r') as f:
                print(f.read())
            print("=== Assembly error ===")
            print(result_asm.stderr)
            raise Exception(f"Assembly failed: {result_asm.stderr}")

        result = subprocess.run(
            ['riscv64-unknown-linux-gnu-objdump', '-d', '-M', 'no-aliases', temp_o_name],
            capture_output=True,
            text=True,
            check=True
        )

        machine_codes = []
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line and ':' in line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    machine_code = parts[1].strip()
                    if machine_code:
                        machine_codes.append(machine_code)

        return ' '.join(machine_codes)

    finally:
        if os.path.exists(temp_s_name):
            os.remove(temp_s_name)
        if os.path.exists(temp_o_name):
            os.remove(temp_o_name)


def match_samples(csv_file_path, json_file_path,
                  sample_method='top', sort_by='loss', ascending=True):
    try:
        df = pd.read_csv(csv_file_path)

    except Exception as e:
        return None, None, None

    total_samples = len(df)
    top_count = max(1, int(total_samples))

    if sample_method == 'top':
        df_selected = df.head(top_count)
    elif sample_method == 'sort':
        if sort_by not in df.columns:
            return None, None, None
        df_selected = df.sort_values(by=sort_by, ascending=ascending).head(top_count)
    else:
        return None, None, None

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        return None, None, None

    json_index = {item['idx']: item for item in json_data}

    matched_results = []
    not_found_indices = []

    length_loss_dict = defaultdict(list)

    for _, row in df_selected.iterrows():
        idx = row['idx']

        if idx in json_index:
            json_sample = json_index[idx]

            block_length = len(json_sample['instructions'])
            loss_value = row['loss']

            length_loss_dict[block_length].append(loss_value)

            matched_sample = {
                'idx': idx,
                'loss': loss_value,
                'measured': row['measured'],
                'prediction': row['prediction'],
                'instructions': json_sample['instructions']
            }
            matched_results.append(matched_sample)
        else:
            not_found_indices.append(idx)

    if not_found_indices:
        print(f"Number of samples with no match found: {len(not_found_indices):,}")

    length_avg_loss = {}
    for length, losses in length_loss_dict.items():
        avg_loss = sum(losses) / len(losses)
        length_avg_loss[length] = avg_loss

    filtered_dict = {length: avg_loss for length, avg_loss in length_avg_loss.items() if length <= 32}
    sorted_items = sorted(filtered_dict.items(), key=lambda x: x[1], reverse=True)
    top_dict = dict(sorted_items[:15])

    return matched_results, top_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BADA')
    parser.add_argument("--exp", type=str, required=True, help="Experiment name")
    parser.add_argument("--epoch", type=int, required=True, help="Number of epochs")
    parser.add_argument("--train", type=str, required=True, help="Path to training data JSON file")
    parser.add_argument("--n_candidates", type=int, default=20000, help="Number of candidate features")
    parser.add_argument("--n_selected_features", type=int, default=100, help="Number of selected feature vectors")
    parser.add_argument("--variants_per_feature", type=int, default=20, help="Number of variants per feature vector")
    parser.add_argument("--max_training_samples", type=int, default=800000, help="Maximum number of training samples")
    parser.add_argument("--hardware", action='store_true', help="Use hardware to generate data")

    args = parser.parse_args()

    csv_file = f'../experiments/{args.exp}/statistics/train_loss_stats_epoch_{args.epoch}.csv'
    json_file = args.train

    start_time = time.time()
    # match_samples(csv_file, json_file, sample_method='top')
    print(f"csv_file: {csv_file}")
    print(f"json_file: {json_file}")
    augmented_dataset = workflow(
        exp_name=args.exp,
        csv_file_path=csv_file,
        json_file_path=json_file,
        n_candidates=args.n_candidates,
        n_selected_features=args.n_selected_features,
        variants_per_feature=args.variants_per_feature,
        max_training_samples=args.max_training_samples,
        hardware=args.hardware
    )

    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds ({total_time / 60:.1f} minutes)")
