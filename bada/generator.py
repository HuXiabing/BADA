"""
Math-based optimized precise basic block generator - 57-dim version

Input: 57-dim feature vector [54 specific instruction type distributions + WAW + RAW + WAR]
Output: Single precisely matched basic block sample
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
import numpy as np
import random
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from bada.analyzer import DataDependencyAnalyzer

@dataclass
class DependencyPlan:
    waw_groups: List[List[int]]
    raw_pairs: List[Tuple[int, int]]
    war_pairs: List[Tuple[int, int]]
    reserved_registers: Dict[str, List[int]]

@dataclass
class RegisterGroup:
    register_name: str
    instruction_indices: List[int]
    waw_count: int
    raw_count: int
    war_count: int

class OptimizedPreciseBasicBlockGenerator:

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

        self.all_registers = (
                ['zero'] +  # x0
                ['ra'] +  # x1
                ['sp'] +  # x2
                ['gp'] +  # x3
                ['tp'] +  # x4
                [f't{i}' for i in range(3)] +  # x5-x7
                ['fp', 's1'] +  # x8-x9
                [f'a{i}' for i in range(8)] +  # x10-x17
                [f's{i}' for i in range(2, 12)] +  # x18-x27
                [f't{i}' for i in range(3, 7)]  # x28-x31
        )

        self.writable_registers = [reg for reg in self.all_registers if reg not in ['zero', 'sp', 'ra', 'gp', 'tp']]
        self.dependency_registers = [f't{i}' for i in range(7)] + [f'a{i}' for i in range(8)] + [f's{i}' for i in
                                                                                                 range(1, 12)]

    def generate_precise_basic_block(self, feature_vector: np.ndarray,
                                     verbose: bool = False) -> List[str]:

        instruction_counts = feature_vector[:54].astype(int)
        total_instructions = sum(instruction_counts)

        target_waw = int(feature_vector[54])
        target_raw = int(feature_vector[55])
        target_war = int(feature_vector[56])

        # Step 1
        dependency_plan = self._plan_dependencies_with_math_optimization(
            total_instructions, target_waw, target_raw, target_war, verbose=0
        )

        # Step 2
        instruction_skeletons = self._generate_instruction_skeletons(instruction_counts)

        # Step 3
        instructions = self._assign_registers_with_optimization(instruction_skeletons, dependency_plan, verbose=0)

        # Step 4
        if verbose:
            analyzer = DataDependencyAnalyzer()
            actual_deps = analyzer.analyze_dependencies(instructions)
            print(f"WAW={actual_deps['waw']}, RAW={actual_deps['raw']}, WAR={actual_deps['war']}")

        return instructions

    def _plan_dependencies_with_math_optimization(self, total_instructions: int, target_waw: int,
                                                  target_raw: int, target_war: int,
                                                  verbose: bool = False) -> DependencyPlan:

        waw_groups = []
        raw_pairs = []
        war_pairs = []
        reserved_registers = {}

        available_dependency_registers = self.dependency_registers.copy()
        random.shuffle(available_dependency_registers)
        register_iter = iter(available_dependency_registers)

        used_instructions = set()

        current_waw = 0
        current_raw = 0
        current_war = 0

        if verbose:
            print(f"Target: WAW={target_waw}, RAW={target_raw}, WAR={target_war}")

        register_groups = []
        iteration_count = 0
        max_iterations = 10

        while (current_waw < target_waw or current_raw < target_raw or current_war < target_war) and \
                len(used_instructions) < total_instructions and iteration_count < max_iterations:

            iteration_count += 1
            remaining_waw = target_waw - current_waw
            remaining_raw = target_raw - current_raw
            remaining_war = target_war - current_war

            best_n = self._find_optimal_n(remaining_waw, remaining_raw, remaining_war,
                                          total_instructions - len(used_instructions))

            if best_n >= 2:
                available_instructions = [i for i in range(total_instructions) if i not in used_instructions]
                if len(available_instructions) >= best_n:
                    selected_instructions = sorted(available_instructions[:best_n])

                    try:
                        register_name = next(register_iter)
                    except StopIteration:
                        available_dependency_registers = self.dependency_registers.copy()
                        random.shuffle(available_dependency_registers)
                        register_iter = iter(available_dependency_registers)
                        register_name = next(register_iter)

                    n = len(selected_instructions)
                    waw_produced = n * (n - 1) // 2
                    raw_produced = n - 1
                    war_produced = n * (n - 1) // 2

                    waw_excess = max(0, current_waw + waw_produced - target_waw)
                    raw_excess = max(0, current_raw + raw_produced - target_raw)
                    war_excess = max(0, current_war + war_produced - target_war)

                    if waw_excess + raw_excess + war_excess > 3:
                        break

                    register_group = RegisterGroup(
                        register_name=register_name,
                        instruction_indices=selected_instructions,
                        waw_count=waw_produced,
                        raw_count=raw_produced,
                        war_count=war_produced
                    )
                    register_groups.append(register_group)
                    waw_groups.append(selected_instructions)

                    for i in range(len(selected_instructions) - 1):
                        raw_pairs.append((selected_instructions[i], selected_instructions[i + 1]))

                    for i in range(len(selected_instructions)):
                        for j in range(i + 1, len(selected_instructions)):
                            war_pairs.append((selected_instructions[i], selected_instructions[j]))

                    reserved_registers[register_name] = selected_instructions
                    current_waw += waw_produced
                    current_raw += raw_produced
                    current_war += war_produced

                    used_instructions.update(selected_instructions)
                else:
                    break
            else:
                break

        remaining_waw = target_waw - current_waw
        remaining_raw = target_raw - current_raw
        remaining_war = target_war - current_war

        if verbose:
            print(f"Remaining: WAW={remaining_waw}, RAW={remaining_raw}, WAR={remaining_war}")

        for i in range(remaining_waw):
            available_instructions = [j for j in range(total_instructions) if j not in used_instructions]
            if len(available_instructions) >= 2:
                pair = sorted(available_instructions[:2])
                waw_groups.append(pair)
                try:
                    waw_register = next(register_iter)
                except StopIteration:
                    available_dependency_registers = self.dependency_registers.copy()
                    random.shuffle(available_dependency_registers)
                    register_iter = iter(available_dependency_registers)
                    waw_register = next(register_iter)

                reserved_registers[waw_register] = pair
                used_instructions.update(pair)

        for i in range(remaining_raw):
            available_instructions = [j for j in range(total_instructions) if j not in used_instructions]
            if len(available_instructions) >= 2:
                write_inst, read_inst = sorted(available_instructions[:2])
                raw_pairs.append((write_inst, read_inst))

                try:
                    raw_register = next(register_iter)
                except StopIteration:
                    available_dependency_registers = self.dependency_registers.copy()
                    random.shuffle(available_dependency_registers)
                    register_iter = iter(available_dependency_registers)
                    raw_register = next(register_iter)

                reserved_registers[raw_register] = [write_inst, read_inst]
                used_instructions.update([write_inst, read_inst])

        for i in range(remaining_war):
            available_instructions = [j for j in range(total_instructions) if j not in used_instructions]
            if len(available_instructions) >= 2:
                read_inst, write_inst = sorted(available_instructions[:2])
                war_pairs.append((read_inst, write_inst))

                try:
                    war_register = next(register_iter)
                except StopIteration:
                    available_dependency_registers = self.dependency_registers.copy()
                    random.shuffle(available_dependency_registers)
                    register_iter = iter(available_dependency_registers)
                    war_register = next(register_iter)

                reserved_registers[war_register] = [read_inst, write_inst]
                used_instructions.update([read_inst, write_inst])

        return DependencyPlan(waw_groups, raw_pairs, war_pairs, reserved_registers)

    def _find_optimal_n(self, remaining_waw: int, remaining_raw: int, remaining_war: int,
                        available_instructions: int) -> int:

        total_remaining = remaining_waw + remaining_raw + remaining_war
        if total_remaining <= 2:
            return 1

        best_n = 1
        best_score = -1

        max_n = min(available_instructions, 8)

        for n in range(2, max_n + 1):
            waw_produced = n * (n - 1) // 2
            raw_produced = n - 1
            war_produced = n * (n - 1) // 2

            waw_satisfied = min(waw_produced, remaining_waw)
            raw_satisfied = min(raw_produced, remaining_raw)
            war_satisfied = min(war_produced, remaining_war)
            total_satisfied = waw_satisfied + raw_satisfied + war_satisfied

            waw_waste = max(0, waw_produced - remaining_waw)
            raw_waste = max(0, raw_produced - remaining_raw)
            war_waste = max(0, war_produced - remaining_war)
            total_waste = waw_waste + raw_waste + war_waste

            waw_waste_ratio = waw_waste / max(remaining_waw, 1) if remaining_waw > 0 else 0
            raw_waste_ratio = raw_waste / max(remaining_raw, 1) if remaining_raw > 0 else 0
            war_waste_ratio = war_waste / max(remaining_war, 1) if remaining_war > 0 else 0

            if waw_waste_ratio > 0.5 or raw_waste_ratio > 0.5 or war_waste_ratio > 0.5:
                continue

            if total_waste > 5:
                continue

            base_score = total_satisfied - total_waste * 1.2
            bonus = 0

            if (waw_satisfied == remaining_waw and waw_waste == 0) or \
                    (raw_satisfied == remaining_raw and raw_waste == 0) or \
                    (war_satisfied == remaining_war and war_waste == 0):
                bonus += 3

            if total_waste == 0:
                bonus += 3
            elif total_waste <= 1:
                bonus += 2
            elif total_waste <= 2:
                bonus += 1

            if total_satisfied >= 3 and max(waw_waste_ratio, raw_waste_ratio, war_waste_ratio) <= 0.2:
                bonus += 2

            if n <= 4:
                bonus += 1
            elif n >= 6:
                bonus -= 1

            final_score = base_score + bonus

            if final_score > best_score:
                best_score = final_score
                best_n = n

        if best_score < 2:
            return 1

        return best_n

    def _generate_instruction_skeletons(self, instruction_counts: np.ndarray) -> List[Dict]:

        skeletons = []
        for i, count in enumerate(instruction_counts):
            if count <= 0:
                continue

            inst_name = self.all_instruction_types[i]
            for _ in range(int(count)):
                skeleton = self._create_instruction_skeleton(inst_name)
                skeletons.append(skeleton)

        random.shuffle(skeletons)
        for i, skeleton in enumerate(skeletons):
            skeleton['index'] = i

        return skeletons

    def _create_instruction_skeleton(self, inst_name: str) -> Dict:

        skeleton = {
            'inst_name': inst_name,
            'dest_regs': [],
            'src_regs': [],
            'imm': None,
            'format': None
        }

        if inst_name in ['slli', 'slliw', 'srai', 'sraiw', 'srli', 'srliw']:
            skeleton['format'] = 'I'
            skeleton['imm'] = random.randint(0, 31)
            skeleton['dest_regs'] = ['none']
            skeleton['src_regs'] = ['none']
        elif inst_name in ['addi', 'addiw', 'slti', 'sltiu']:
            skeleton['format'] = 'I'
            skeleton['imm'] = random.choice([-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16])
            skeleton['dest_regs'] = ['none']
            skeleton['src_regs'] = ['none']
        elif inst_name in ['andi', 'ori', 'xori']:
            skeleton['format'] = 'I'
            skeleton['imm'] = random.choice([0, 1, 2, 4, 8, 15, 16, 31])
            skeleton['dest_regs'] = ['none']
            skeleton['src_regs'] = ['none']
        elif inst_name in ['auipc', 'lui']:
            skeleton['format'] = 'U'
            skeleton['imm'] = random.choice([0x0, 0x1, 0x10, 0x100, 0x1000])
            skeleton['dest_regs'] = ['none']
            skeleton['src_regs'] = []
        elif inst_name in ['lb', 'lbu', 'ld', 'lh', 'lhu', 'lw', 'lwu']:
            skeleton['format'] = 'I'
            skeleton['imm'] = random.choice([-64, -32, -16, -8, -4, 0, 4, 8, 16, 32, 64])
            skeleton['dest_regs'] = ['none']
            skeleton['src_regs'] = ['none']
        elif inst_name in ['sb', 'sd', 'sh', 'sw']:
            skeleton['format'] = 'S'
            skeleton['imm'] = random.choice([-64, -32, -16, -8, -4, 0, 4, 8, 16, 32, 64])
            skeleton['dest_regs'] = []
            skeleton['src_regs'] = ['none', 'none']
        else:
            skeleton['format'] = 'R'
            skeleton['dest_regs'] = ['none']
            skeleton['src_regs'] = ['none', 'none']

        return skeleton

    def _assign_registers_with_optimization(self, skeletons: List[Dict],
                                            dependency_plan: DependencyPlan,
                                            verbose: bool = False) -> List[str]:

        register_assignments = {}
        for skeleton in skeletons:
            idx = skeleton['index']
            register_assignments[idx] = {
                'dest': ['none'] * len(skeleton['dest_regs']),
                'src': ['none'] * len(skeleton['src_regs'])
            }

        instruction_roles = defaultdict(list)

        for waw_idx, waw_group in enumerate(dependency_plan.waw_groups):
            for reg_name, instructions in dependency_plan.reserved_registers.items():
                if set(instructions) == set(waw_group):
                    for inst_idx in waw_group:
                        instruction_roles[inst_idx].append(('waw_dest', reg_name))
                    break

        for raw_idx, (write_inst, read_inst) in enumerate(dependency_plan.raw_pairs):

            raw_register = None
            for reg_name, instructions in dependency_plan.reserved_registers.items():
                if write_inst in instructions and read_inst in instructions:
                    raw_register = reg_name
                    break

            if raw_register:
                instruction_roles[write_inst].append(('raw_write', raw_register))
                instruction_roles[read_inst].append(('raw_read', raw_register))

        for war_idx, (read_inst, write_inst) in enumerate(dependency_plan.war_pairs):

            war_register = None
            for reg_name, instructions in dependency_plan.reserved_registers.items():
                if read_inst in instructions and write_inst in instructions:
                    war_register = reg_name
                    break

            if war_register:
                instruction_roles[read_inst].append(('war_read', war_register))
                instruction_roles[write_inst].append(('war_write', war_register))

        used_registers = set()
        for roles in instruction_roles.values():
            for _, reg in roles:
                used_registers.add(reg)

        free_registers = [reg for reg in self.writable_registers if reg not in used_registers]
        random.shuffle(free_registers)

        if free_registers:
            free_register_cycle = iter(free_registers * 10)
        else:
            free_register_cycle = iter(self.writable_registers * 10)

        for skeleton in skeletons:
            idx = skeleton['index']
            roles = instruction_roles[idx]

            dest_role_regs = [reg for role, reg in roles if role in ['waw_dest', 'raw_write', 'war_write']]
            for i in range(len(dest_role_regs)):
                if i < len(register_assignments[idx]['dest']):
                    register_assignments[idx]['dest'][i] = dest_role_regs[i]

            src_role_regs = [reg for role, reg in roles if role in ['raw_read', 'war_read']]
            assigned_src_regs = set()
            src_role_idx = 0

            for i in range(len(register_assignments[idx]['src'])):
                if src_role_idx < len(src_role_regs):
                    candidate_reg = src_role_regs[src_role_idx]

                    if candidate_reg not in assigned_src_regs:
                        register_assignments[idx]['src'][i] = candidate_reg
                        assigned_src_regs.add(candidate_reg)
                        src_role_idx += 1
                    else:
                        src_role_idx += 1
                else:
                    break

        for skeleton in skeletons:
            idx = skeleton['index']

            for i in range(len(register_assignments[idx]['dest'])):
                if register_assignments[idx]['dest'][i] == 'none':
                    try:
                        register_assignments[idx]['dest'][i] = next(free_register_cycle)
                    except StopIteration:
                        register_assignments[idx]['dest'][i] = random.choice(self.writable_registers)

            current_src_regs = set()

            for i in range(len(register_assignments[idx]['src'])):
                if register_assignments[idx]['src'][i] == 'none':
                    attempts = 0
                    max_attempts = 50
                    while attempts < max_attempts:
                        try:
                            candidate_reg = next(free_register_cycle)
                        except StopIteration:
                            candidate_reg = random.choice(self.all_registers)

                        if candidate_reg not in current_src_regs:
                            register_assignments[idx]['src'][i] = candidate_reg
                            current_src_regs.add(candidate_reg)
                            break

                        attempts += 1

                    if attempts >= max_attempts:
                        register_assignments[idx]['src'][i] = candidate_reg
                        current_src_regs.add(candidate_reg)
                else:
                    current_src_regs.add(register_assignments[idx]['src'][i])

        final_instructions = []
        for skeleton in skeletons:
            idx = skeleton['index']
            instruction = self._skeleton_to_instruction(skeleton, register_assignments[idx])
            final_instructions.append(instruction)

        return final_instructions

    def _skeleton_to_instruction(self, skeleton: Dict, register_assignment: Dict) -> str:

        inst_name = skeleton['inst_name']
        dest_regs = register_assignment['dest']
        src_regs = register_assignment['src']

        def safe_register(reg, is_dest=True):
            if reg == 'none':
                if is_dest:
                    return random.choice(self.writable_registers)
                else:
                    return random.choice(self.all_registers)
            return reg

        if skeleton['format'] == 'R':
            rd = safe_register(dest_regs[0] if dest_regs else 'none', True)
            rs1 = safe_register(src_regs[0] if len(src_regs) > 0 else 'none', False)
            rs2 = safe_register(src_regs[1] if len(src_regs) > 1 else 'none', False)
            return f"{inst_name}\t{rd},{rs1},{rs2}"
        elif skeleton['format'] == 'I':
            if inst_name in ['lb', 'lbu', 'ld', 'lh', 'lhu', 'lw', 'lwu']:
                rd = safe_register(dest_regs[0] if dest_regs else 'none', True)
                rs1 = safe_register(src_regs[0] if src_regs else 'none', False)
                offset = skeleton['imm']
                return f"{inst_name}\t{rd},{offset}({rs1})"
            else:
                rd = safe_register(dest_regs[0] if dest_regs else 'none', True)
                rs1 = safe_register(src_regs[0] if src_regs else 'none', False)
                imm = skeleton['imm']
                return f"{inst_name}\t{rd},{rs1},{imm}"
        elif skeleton['format'] == 'S':
            rs1 = safe_register(src_regs[1] if len(src_regs) > 1 else 'none', False)
            rs2 = safe_register(src_regs[0] if len(src_regs) > 0 else 'none', False)
            offset = skeleton['imm']
            return f"{inst_name}\t{rs2},{offset}({rs1})"
        elif skeleton['format'] == 'U':
            rd = safe_register(dest_regs[0] if dest_regs else 'none', True)
            imm = skeleton['imm']
            return f"{inst_name}\t{rd},0x{imm:x}"

        return inst_name


def demo_optimized_usage():

    generator = OptimizedPreciseBasicBlockGenerator()
    # ['sll', 'sllw', 'sra', 'sraw', 'srl', 'srlw', 'slli', 'slliw', 'srai', 'sraiw',
    # 'srli', 'srliw', 'add', 'addw', 'sub', 'subw', 'addi', 'addiw', 'auipc', 'lui',
    # 'and', 'xor', 'or', 'andi', 'ori', 'xori', 'slt', 'sltu', 'slti', 'sltiu',
    # 'mul', 'mulh', 'mulhsu', 'mulhu', 'mulw', 'div', 'divu', 'divuw', 'divw', 'rem',
    # 'remu', 'remuw', 'remw', 'lb', 'lbu', 'ld', 'lh', 'lhu', 'lw', 'lwu', 'sb', 'sd', 'sh', 'sw']

    feature_vector1 = np.zeros(57)
    feature_vector1[0] = 1  # sll
    feature_vector1[10] = 1  # srli
    feature_vector1[20] = 2  # and
    feature_vector1[30] = 1  # mul
    feature_vector1[54] = 2  # WAW
    feature_vector1[55] = 3  # RAW
    feature_vector1[56] = 1  # WAR

    sample1 = generator.generate_precise_basic_block(feature_vector1, verbose=1)

    for i, inst in enumerate(sample1):
        print(f"{i}: {inst}")


if __name__ == "__main__":
    demo_optimized_usage()