import re
import torch
import os
from typing import List, Dict, Any, Optional
import json
class RISCVTokenizer:
    """RISC-V tokenizer with custom tokenization and vocabulary"""

    def __init__(self,
                 vocab_size: int = 300,
                 max_instr_length: int = 8):
        """
        Args:
            vocab_size: the upper limit of vocabulary size (not actually used, reserved for compatibility)
            max_instr_length: maximum number of tokens for each instruction, fixed to 8
        """
        self.vocab_size = vocab_size
        self.max_instr_length = max_instr_length

        self._load_predefined_dict()
        self.implicit_list = ['fld', 'flw', 'fsd', 'fsw', 'lb', 'lbu', 'ld', 'lh', 'lhu', 'lw',
                              'lwu', 'sb', 'sd', 'sh', 'sw']

    def _load_predefined_dict(self):
        """ load predefined vocabulary """
        dict_path = os.path.join(os.path.dirname(__file__), 'vocab.dump')
        if os.path.exists(dict_path):
            self.vocab = torch.load(dict_path)
            self.inverse_vocab = torch.load("data/inverse_vocab.dump")
        else:
            # if the predefined dict does not exist, build it manually
            self._build_vocab_manually()

        # inverse the vocabulary to decode token IDs
        # self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.current_vocab_size = len(self.vocab)

    def _build_vocab_manually(self):

        # register_aliases = {
        #     'zero': 'x0', 'ra': 'x1', 'sp': 'x2', 'gp': 'x3', 'tp': 'x4',
        #     't0': 'x5', 't1': 'x6', 't2': 'x7', 's0': 'x8', 'fp': 'x8',
        #     's1': 'x9', 'a0': 'x10', 'a1': 'x11', 'a2': 'x12', 'a3': 'x13',
        #     'a4': 'x14', 'a5': 'x15', 'a6': 'x16', 'a7': 'x17',
        #     's2': 'x18', 's3': 'x19', 's4': 'x20', 's5': 'x21', 's6': 'x22',
        #     's7': 'x23', 's8': 'x24', 's9': 'x25', 's10': 'x26', 's11': 'x27',
        #     't3': 'x28', 't4': 'x29', 't5': 'x30', 't6': 'x31'
        # }

        special_token = {'<PAD>': 0, '<BLOCK_START>': 1, '<BLOCK_END>': 2, '<ADDRESS>': 3,
                         '<E>': 4, '<D>': 5, '<S>': 6, '<CONST>': 7, '<CSR>': 8}

        _xregs_abi_list = ['zero', 'ra', 'sp', 'gp', 'tp', 't0', 't1', 't2',
                      's0', 's1', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5',
                      'a6', 'a7', 's2', 's3', 's4', 's5', 's6', 's7',
                      's8', 's9', 's10', 's11', 't3', 't4', 't5', 't6']

        _fregs_abi_list = ['ft0', 'ft1', 'ft2', 'ft3', 'ft4', 'ft5', 'ft6', 'ft7',
                      'fs0', 'fs1', 'fa0', 'fa1', 'fa2', 'fa3', 'fa4', 'fa5',
                      'fa6', 'fa7', 'fs2', 'fs3', 'fs4', 'fs5', 'fs6', 'fs7',
                      'fs8', 'fs9', 'fs10', 'fs11', 'ft8', 'ft9', 'ft10', 'ft11']

        _xregs_numeric_list = [f'x{i}' for i in range(32)]
        _fregs_numeric_list = [f'f{i}' for i in range(32)]

        _xregs_numeric = {_xregs_numeric_list[i]: i + 9 for i in range(len(_xregs_numeric_list))}
        _xregs_abi = { _xregs_abi_list[i] : i + 9 for i in range(len(_xregs_abi_list))}
        _fregs_numeric = {_fregs_numeric_list[i]: i + 41 for i in range(len(_fregs_numeric_list))}
        _fregs_abi = { _fregs_abi_list[i] : i + 41 for i in range(len(_fregs_abi_list))}

        _rv64I_list = ["add", "addi", "addiw", "addw", "and", "andi", "auipc", "beq", "bge",
                    "bgeu", "blt", "bltu", "bne", "ebreak", "ecall", "fence", "jal", "jalr",
                    "lb", "lbu", "ld", "lh", "lhu", "lui", "lw", "lwu", "or", "ori", "sb",
                    "sd", "sh", "sll", "slli", "slliw", "sllw", "slt", "slti", "sltiu",
                    "sltu", "sra", "srai", "sraiw", "sraw", "srl", "srli", "srliw", "srlw",
                    "sub", "subw", "sw", "xor", "xori"] #52
        _rv64I = {_rv64I_list[i]: i + 73 for i in range(len(_rv64I_list))}

        _rv64M_list = ["div", "divu", "divuw", "divw", "mul", "mulh", "mulhsu", "mulhu", "mulw",
                       "rem", "remu", "remuw", "remw"]  # 13
        _rv64M = {_rv64M_list[i]: i + 125 for i in range(len(_rv64M_list))}

        # _rv64A_list =["amoadd.d", "amoadd.w", "amoand.d", "amoand.w", "amomax.d", "amomax.w",
        #             "amomaxu.d", "amomaxu.w", "amomin.d", "amomin.w", "amominu.d", "amominu.w",
        #             "amoor.d", "amoor.w", "amoswap.d", "amoswap.w", "amoxor.d", "amoxor.w",
        #             "lr.d", "lr.w", "sc.d", "sc.w"] #22
        # _rv64A = {_rv64A_list[i]: i + 165 for i in range(len(_rv64A_list))}


        # _rv64FD_list = ["fadd.d", "fadd.s", "fclass.d", "fclass.s", "fcvt.d.l", "fcvt.d.lu", "fcvt.d.s",
        #             "fcvt.d.w", "fcvt.d.wu", "fcvt.l.d", "fcvt.l.s", "fcvt.lu.d", "fcvt.lu.s",
        #             "fcvt.s.d", "fcvt.s.l", "fcvt.s.lu", "fcvt.s.w", "fcvt.s.wu", "fcvt.w.d",
        #             "fcvt.w.s", "fcvt.wu.d", "fcvt.wu.s", "fdiv.d", "fdiv.s", "feq.d", "feq.s",
        #             "fld", "fle.d", "fle.s", "flt.d", "flt.s", "flw", "fmadd.d", "fmadd.s",
        #             "fmax.d", "fmax.s", "fmin.d", "fmin.s", "fmsub.d", "fmsub.s", "fmul.d",
        #             "fmul.s", "fmv.d.x", "fmv.w.x", "fmv.x.d", "fmv.x.w", "fnmadd.d", "fnmadd.s",
        #             "fnmsub.d", "fnmsub.s", "fsd", "fsgnj.d", "fsgnj.s", "fsgnjn.d", "fsgnjn.s",
        #             "fsgnjx.d", "fsgnjx.s", "fsqrt.d", "fsqrt.s", "fsub.d", "fsub.s", "fsw"] #62
        # _rv64FD = {_rv64FD_list[i]: i + 160 for i in range(len(_rv64FD_list))}
        #
        # _CSR_list = ["csrrc", "csrrci", "csrrs", "csrrsi", "csrrw", "csrrwi", "fence.i"] #7
        # _CSR = {_CSR_list[i]: i + 222 for i in range(len(_CSR_list))}

        # _rv64 = {**_rv64A, **_rv64M, **_rv64I, **_rv64FD, **_CSR}
        _rv64 = {**_rv64M, **_rv64I}

        self.vocab ={**special_token, **_xregs_abi, **_fregs_abi, **_xregs_numeric, **_fregs_numeric, **_rv64}
        self.inverse_vocab = {v: k for k, v in {**special_token, **_xregs_abi, **_fregs_abi, **_rv64}.items()}

        torch.save(self.vocab, "data/vocab.dump")
        torch.save(self.inverse_vocab, "data/inverse_vocab.dump")
        """
        {'<PAD>': 0, '<BLOCK_START>': 1, '<BLOCK_END>': 2, '<ADDRESS>': 3, '<E>': 4, '<D>': 5, '<S>': 6, '<CONST>': 7, '<CSR>': 8, 
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
 #73-137
        """

    def separate_disp_and_reg(self, s):

        match = re.match(r'-?\d+\((\w+)\)', s)
        if match:
            full_match = match.group(0)
            reg = match.group(1)
            disp_str = full_match[:-(len(reg) + 2)]
            disp = int(disp_str)
            return disp, reg
        else:
            raise ValueError("The string does not match the expected format.")

    def is_numeric(self, s):

        pattern = re.compile(r'^-?\d+(\.\d+)?$')
        return bool(pattern.match(s))

    def is_address_string(self, s):

        pattern = re.compile(r'^0x[0-9a-fA-F]+$')
        return bool(pattern.match(s))

    def zero_reg(self, instr):

        return [instr[0], '<E>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']

    def one_reg_num(self, instr):

        pattern = r'^-?\d+$'
        if re.match(pattern, instr[1]):  # "j, -1666"
            return [instr[0], '<CONST>', '<E>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']
        else:
            return [instr[0], instr[1], '<E>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']

    def two_reg(self, instr):

        if instr[0] in self.implicit_list:
            _, rs = self.separate_disp_and_reg(instr[-1])
            return [instr[0], '<D>', instr[1], '<S>', rs, '<CONST>', '<E>', '<PAD>']
        elif self.is_numeric(instr[-1]):
            return [instr[0], '<D>', instr[1], '<S>', '<CONST>', '<E>', '<PAD>', '<PAD>']
        elif self.is_address_string(instr[-1]):
            return [instr[0], '<D>', instr[1], '<S>', '<ADDRESS>', '<E>', '<PAD>', '<PAD>']
        else:
            return [instr[0], '<D>', instr[1], '<S>', instr[2], '<E>', '<PAD>', '<PAD>']

    def three_reg(self, instr):

        if instr[1] == 'csr':
            if self.is_numeric(instr[-1]):
                return [instr[0], '<D>', instr[1], '<S>', '<CSR>', '<CONST>', '<E>', '<PAD>']
            elif self.is_address_string(instr[-1]):
                return [instr[0], '<D>', instr[1], '<S>', '<CSR>', '<ADDRESS>', '<E>', '<PAD>']
            else:
                return [instr[0], '<D>', instr[1], '<S>', '<CSR>', instr[-1], '<E>', '<PAD>']
        elif self.is_numeric(instr[-1]):
            return [instr[0], '<D>', instr[1], '<S>', instr[2], '<CONST>', '<E>', '<PAD>']
        elif self.is_address_string(instr[-1]):
            return [instr[0], '<D>', instr[1], '<S>', instr[2], '<ADDRESS>', '<E>', '<PAD>']
        else:
            return [instr[0], '<D>', instr[1], '<S>', instr[2], instr[3], '<E>', '<PAD>']

    def four_reg(self, instr):

        return [instr[0], '<D>', instr[1], '<S>', instr[2], instr[3], instr[4], '<E>']

    def rformat(self, instr):
        # and x12, x26, x26
        return [instr[0], '<D>', instr[1], '<S>', instr[2], instr[3], '<E>', '<PAD>']

    def extract_register_parts(self, text):
        """
        8(x26) ->  (number, register) or None if the pattern doesn't match
        """
        pattern = r'(\d+)\(([^)]+)\)'
        match = re.search(pattern, text)

        if match:
            return match.group(1), match.group(2)
        else:
            return None

    def iformat(self, instr):
        # andi x3, x5, 8
        return [instr[0], '<D>', instr[1], '<S>', instr[2], '<CONST>', '<E>', '<PAD>']

    def uformat(self, instr):
        # auipc x9, 0x8
        return [instr[0], '<D>', instr[1], '<S>', '<CONST>', '<E>', '<PAD>', '<PAD>']

    def loadformat(self, instr):
        # ld x30, 8(x26)
        _, reg = self.extract_register_parts(instr[-1])
        return [instr[0], '<D>', instr[1], '<S>', '<ADDRESS>', reg, '<E>', '<PAD>']

    def storeformat(self, instr):
        # sd x30, 8(x26)
        _, reg = self.extract_register_parts(instr[-1])

        return [instr[0], '<D>', '<ADDRESS>', reg, '<S>', instr[1],  '<E>','<PAD>']


    def tokenize_instruction(self, instruction: str) -> List[str]:
        """
        transform RISC-V instruction into token list,
        eg: "add a5,s1,a0" -> ['add', '<D>', 'a5', '<S>', 's1', 'a0', '<E>', '<PAD>']

        Args:
            instruction: "add a5,s1,a0"

        Returns:
            ['add', '<D>', 'a5', '<S>', 's1', 'a0', '<E>', '<PAD>']
        """
        pattern = r'[^ ,\t]+'
        instr = re.findall(pattern, instruction)
        if instr[0] in ['add', 'addw', 'and', 'or',
              'div', 'divu', 'divuw', 'divw','mul', 'mulh', 'mulhsu', 'mulhu', 'mulw', 'rem', 'remu','remuw', 'remw',
              'sll', 'sllw', 'slt', 'sltu', 'sra',
              'sraw', 'srl', 'srlw', 'sub', 'subw', 'xor']:  # 28
            tokenized = self.rformat(instr)
        elif instr[0] in ['addi', 'addiw', 'andi', 'ori', 'slli', 'slliw',
              'slti', 'sltiu', 'srai', 'sraiw', 'srli', 'srliw', 'xori']:  # 13
            tokenized = self.iformat(instr)
        elif instr[0] in ['auipc', 'lui']:  #2
            tokenized = self.uformat(instr)
        elif instr[0] in ['lb', 'lbu', 'ld', 'lh', 'lhu', 'lw', 'lwu']:  #7
            tokenized = self.loadformat(instr)
        elif instr[0] in ['sb', 'sd', 'sh', 'sw']:  #4
            tokenized = self.storeformat(instr)
        elif instr[0] in ['fence', 'ebreak', 'ecall']:
            tokenized = self.zero_reg(instr)
        else:
            print(instr)
            raise ValueError("The string does not match the expected format.")
        return tokenized

    def encode_instruction(self, instruction: str) -> List[int]:
        """
        Encoded the instruction into token ID sequence,
        eg: "add a5,s1,a0" -> [ 72, 5, 24, 6, 18, 19, 4, 0]

        Args:
            instruction: "add a5,s1,a0"

        Returns:
            [ 72, 5, 24, 6, 18, 19, 4, 0]
        """
        tokens = self.tokenize_instruction(instruction)
        encoded = [self.vocab.get(token, self.vocab.get('<PAD>', 0)) for token in tokens]

        return encoded

    def tokenized_bb(self, basic_block: str) -> List[List[str]]:
        """
        Tokenizes a basic block into a list of token lists.
        Args:
            basic_block: A string representing the basic block, with instructions separated by newline characters.

        Returns:
            A list of tokenized lists.
        """
        bb_tokens = []
        pattern = r'[^ ,\t]+'
        bb = [re.findall(pattern, s) for s in basic_block.strip().split("\\n")]

        for instruction in bb:
            tokenized = self.tokenize_instruction(instruction)
            bb_tokens.append(tokenized)

        return bb_tokens

    def decode_instruction(self, instruction: List[int]) -> str:
        """
        Decode the instruction from token ID sequence to instruction string,
        eg: [72, 5, 24, 6, 18, 19, 4, 0] -> ['add', '<D>', 'a5', '<S>', 's1', 'a0', '<E>', '<PAD>']
        """
        decoded = [self.inverse_vocab.get(token, '<PAD>') for token in instruction]
        return decoded

if __name__ == "__main__":
    tokenizer = RISCVTokenizer()
    # bb = ["amoswap.w.rl\ta5,a5,(s1)", "amoswap.d.aqrl  a1,a1,(a0)", "sd\ts1,8(sp)"]
    # for i in bb:
    #     print(tokenizer.tokenize_instruction(i))
    bb = [[ 94,   5,  24,   6,   3,  19,   4,   0],
        [ 97,   5,  22,   6,   3,  19,   4,   0],
        [ 97,   5,  23,   6,   3,  19,   4,   0],
        [ 97,   5,  21,   6,   3,  19,   4,   0],
        [133,   5,  22,   6,  22,  24,   4,   0],
        [ 74,   5,  11,   6,  11,   7,   4,   0],
        [ 74,   5,  25,   6,   9,   7,   4,   0],
        [102,   5,   3,  11,   6,  27,   4,   0],
        [102,   5,   3,  11,   6,  28,   4,   0],
        [102,   5,   3,  11,   6,  29,   4,   0],
        [102,   5,   3,  11,   6,  10,   4,   0],
        [102,   5,   3,  11,   6,  17,   4,   0],
        [102,   5,   3,  11,   6,  18,   4,   0],
        [ 78,   5,  21,   6,  21,   7,   4,   0],
        [133,   5,  24,   6,  23,  24,   4,   0],
        [ 74,   5,  27,   6,  20,   7,   4,   0],
        [128,   5,  28,   6,  22,  25,   4,   0],
        [128,   5,  29,   6,  24,  25,   4,   0]]
    for i in bb:
        print(tokenizer.decode_instruction(i))