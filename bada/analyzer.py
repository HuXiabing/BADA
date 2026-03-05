from collections import defaultdict, Counter, OrderedDict
from typing import List, Dict, Tuple, Any, Set
import re

class DataDependencyAnalyzer:
    """Calculate WAW/RAW/WAR dependencies"""

    def __init__(self):
        self.register_pattern = re.compile(r'\b([a-z]\d+|[a-z]{2,3})\b')

    def extract_registers(self, instruction: str) -> Tuple[List[str], List[str]]:

        instruction = instruction.strip().lower()
        parts = instruction.replace('\t', ' ').split()

        if len(parts) < 2:
            return [], []

        inst_name = parts[0]
        operands = ' '.join(parts[1:]).split(',')

        dest_regs = []
        src_regs = []

        if inst_name in ['lb', 'lbu', 'ld', 'lh', 'lhu', 'lw', 'lwu']:
            if len(operands) >= 2:
                dest_regs.append(operands[0].strip())
                offset_reg = re.search(r'\(([^)]+)\)', operands[1])
                if offset_reg:
                    src_regs.append(offset_reg.group(1).strip())

        elif inst_name in ['sb', 'sd', 'sh', 'sw']:
            if len(operands) >= 2:
                src_regs.append(operands[0].strip())
                offset_reg = re.search(r'\(([^)]+)\)', operands[1])
                if offset_reg:
                    src_regs.append(offset_reg.group(1).strip())

        elif inst_name in ['addi', 'addiw', 'slti', 'sltiu', 'andi', 'ori', 'xori',
                           'slli', 'slliw', 'srai', 'sraiw', 'srli', 'srliw']:
            if len(operands) >= 2:
                dest_regs.append(operands[0].strip())
                src_regs.append(operands[1].strip())

        elif inst_name in ['auipc', 'lui']:
            if len(operands) >= 1:
                dest_regs.append(operands[0].strip())

        else:
            if len(operands) >= 3:
                dest_regs.append(operands[0].strip())
                src_regs.append(operands[1].strip())
                src_regs.append(operands[2].strip())
            elif len(operands) >= 2:
                dest_regs.append(operands[0].strip())
                src_regs.append(operands[1].strip())

        dest_regs = [reg for reg in dest_regs if reg and reg != 'zero']
        src_regs = [reg for reg in src_regs if reg]

        return dest_regs, src_regs

    def analyze_dependencies(self, instructions: List[str]) -> Dict[str, int]:

        dependencies = {'waw': 0, 'raw': 0, 'war': 0}

        register_writes = defaultdict(list)  # register -> [write_instruction_indices]
        register_reads = defaultdict(list)  # register -> [read_instruction_indices]

        for i, instruction in enumerate(instructions):
            dest_regs, src_regs = self.extract_registers(instruction)

            for dest_reg in dest_regs:
                register_writes[dest_reg].append(i)

            for src_reg in src_regs:
                register_reads[src_reg].append(i)

        for i, instruction in enumerate(instructions):
            dest_regs, src_regs = self.extract_registers(instruction)

            for src_reg in src_regs:
                if src_reg in register_writes:
                    prior_writes = [w for w in register_writes[src_reg] if w < i]
                    if prior_writes:
                        dependencies['raw'] += 1

            for dest_reg in dest_regs:
                if dest_reg in register_reads:
                    prior_reads = [r for r in register_reads[dest_reg] if r < i]
                    dependencies['war'] += len(prior_reads)

            for dest_reg in dest_regs:
                if dest_reg in register_writes:
                    prior_writes = [w for w in register_writes[dest_reg] if w < i]
                    dependencies['waw'] += len(prior_writes)

        return dependencies

def test():
    analyzer = DataDependencyAnalyzer()

    test_instructions = []
    deps = analyzer.analyze_dependencies(test_instructions)
    print("length: ",len(test_instructions))
    print(f"waw: {deps['waw']}, raw: {deps['raw']}, war: {deps['war']}")


if __name__ == "__main__":

    test()