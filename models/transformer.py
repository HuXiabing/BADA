# https://github.com/tatp22/multidim-positional-encoding
# pip install positional-encodings[pytorch]
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, Summer
from torch.utils.checkpoint import checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

def method_dummy_wrapper(method):
    @functools.wraps(method)
    def wrapped(dummy, *args, **kwargs):
        return method(*args, **kwargs)
    return wrapped

def get_positional_encoding_1d(dim):
    return Summer(PositionalEncoding1D(dim))

def get_positional_encoding_2d(dim):
    return Summer(PositionalEncoding2D(dim))

class TransformerModel(nn.Module):

    def __init__(self, config):

        super(TransformerModel, self).__init__()
        self.config = config
        self.device = torch.device(config.device)

        self.model = DeepPM(
            dim=config.embed_dim,
            n_heads=config.num_heads,
            dim_ff=config.hidden_dim,
            pad_idx=0,
            vocab_size=config.vocab_size,
            num_basic_block_layer=2,
            num_instruction_layer=2,
            num_op_layer=config.num_layers,
            use_checkpoint=getattr(config, 'use_checkpoint', False),
            use_layernorm=getattr(config, 'use_layernorm', False),
            use_bb_attn=getattr(config, 'use_bb_attn', False),
            use_seq_attn=getattr(config, 'use_seq_attn', False),
            use_op_attn=getattr(config, 'use_op_attn', False),
            use_pos_2d=getattr(config, 'use_pos_2d', False),
            dropout=config.dropout,
            pred_drop=config.dropout,
            activation='gelu',
            handle_neg=getattr(config, 'handle_neg', False)
        )

    def forward(self, x):

        model_input = {
            'x': x['x'],
            'bb_attn_mod': x['bb_attn_mod'],
            'seq_attn_mod': x['seq_attn_mod'],
            'op_attn_mod': x['op_attn_mod']
        }

        output = self.model(model_input)

        return output

    def count_parameters(self) -> int:
        print("dropout",self.config.dropout)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class DeepPM(nn.Module):

    def __init__(self, dim=128, n_heads=2, dim_ff=256,
                 pad_idx=0, vocab_size=256,
                 num_basic_block_layer=1,
                 num_instruction_layer=0,
                 num_op_layer=0, use_checkpoint=False, use_layernorm=False,
                 use_bb_attn=False, use_seq_attn=False, use_op_attn=False,
                 use_pos_2d=False, dropout=None, pred_drop=0.0, activation='gelu', handle_neg=False):
        super().__init__()

        self.num_basic_block_layer = num_basic_block_layer
        self.num_instruction_layer = num_instruction_layer
        self.num_op_layer = num_op_layer
        if self.num_basic_block_layer <= 0:
            raise ValueError('num_basic_block_layer must be larger than 1')

        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            self.dummy = torch.zeros(1, requires_grad=True, device='cuda')
        else:
            self.dummy = None

        self.use_pos_2d = use_pos_2d
        if self.use_pos_2d:
            self.pos_embed_2d = get_positional_encoding_2d(dim)

        self.pos_embed = get_positional_encoding_1d(dim)

        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, dim, self.pad_idx)

        self.basic_block = DeepPMBasicBlock(dim, dim_ff, n_heads, num_basic_block_layer, use_layernorm=use_layernorm,
                                            use_checkpoint=use_checkpoint, dummy=self.dummy, dropout=dropout,
                                            activation=activation,
                                            handle_neg=handle_neg)

        if self.num_instruction_layer > 0:
            self.instruction_block = DeepPMSeq(dim, dim_ff, n_heads, num_instruction_layer, use_layernorm=use_layernorm,
                                               use_checkpoint=use_checkpoint, dummy=self.dummy, dropout=dropout,
                                               activation=activation,
                                               handle_neg=handle_neg)
        if self.num_op_layer > 0:
            self.op_block = DeepPMOp(dim, dim_ff, n_heads, num_op_layer, use_layernorm=use_layernorm,
                                     use_checkpoint=use_checkpoint, dummy=self.dummy, dropout=dropout,
                                     activation=activation,
                                     handle_neg=handle_neg)

        self.prediction = nn.Sequential(
            nn.Dropout(pred_drop),
            nn.Linear(dim, 1)
        )

        self.use_bb_attn = use_bb_attn
        self.use_seq_attn = use_seq_attn
        self.use_op_attn = use_op_attn

    def forward(self, x):
        bb_attn_mod = x['bb_attn_mod'] if self.use_bb_attn else None
        seq_attn_mod = x['seq_attn_mod'] if self.use_seq_attn else None
        op_attn_mod = x['op_attn_mod'] if self.use_op_attn else None
        x = x['x']

        # B I S
        batch_size, inst_size, seq_size = x.shape
        mask = x == self.pad_idx
        op_seq_mask = mask.all(dim=-1)

        #  B I S D
        output = self.embed(x)

        if self.use_pos_2d:
            output = self.pos_embed_2d(output)
        else:
            # B*I S H
            output = output.view(batch_size * inst_size, seq_size, -1)
            output = self.pos_embed(output)

            # B I S H
            output = output.view(batch_size, inst_size, seq_size, -1)

        output = self.basic_block(output, mask, bb_attn_mod)

        if self.num_instruction_layer > 0:
            output = self.instruction_block(output, mask, op_seq_mask, seq_attn_mod)

        # reduce
        # B I H
        output = output[:, :, 0]
        output = self.pos_embed(output)

        if self.num_op_layer > 0:
            output = self.op_block(output, op_seq_mask, op_attn_mod)

        #  B I
        output = output.sum(dim=1)
        output = self.prediction(output).squeeze(1)
        return output

class DeepPMSeq(nn.Module):
    def __init__(self, dim, dim_ff, n_heads, n_layers, dropout=None,
                 use_layernorm=False, layer_norm_eps=1e-05, use_checkpoint=False, activation='gelu', dummy=None,
                 handle_neg=False):
        super().__init__()

        self.tr = nn.ModuleList([DeepPMTransformerEncoderLayer(dim, n_heads, dim_ff,
                                 use_layernorm=use_layernorm, layer_norm_eps=layer_norm_eps,
                                 dropout=dropout, activation=activation, handle_neg=handle_neg) for _ in range(n_layers)])

        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            if dummy is None:
                self.dummy = torch.zeros(1, requires_grad=True, device='cuda')
            else:
                self.dummy = dummy

    def forward(self, x, mask, op_seq_mask, weighted_attn=None):
        """
        x: [batch_size, inst_size, seq_size, dim]
        mask: [batch_size, inst_size, seq_size]
        op_seq_mask: [batch_size, inst_size]
        """

        batch_size, inst_size, seq_size, _ = x.shape

        x = x.view(batch_size * inst_size, seq_size, -1)  # (batch_size * inst_size, seq_size, dim)
        mask = mask.view(batch_size * inst_size, seq_size)  # (batch_size * inst_size, seq_size)
        op_seq_mask = op_seq_mask.view(batch_size * inst_size)  # (batch_size * inst_size)

        x = x.masked_fill(op_seq_mask.unsqueeze(-1).unsqueeze(-1), 1)  # (batch_size * inst_size, seq_size, dim)
        mod_mask = mask.masked_fill(op_seq_mask.unsqueeze(-1), False)  # (batch_size * inst_size, seq_size)

        for block in self.tr:
            if self.use_checkpoint:
                x = checkpoint(method_dummy_wrapper(block), self.dummy, x, mod_mask, weighted_attn)
            else:
                x = block(x, mod_mask, weighted_attn)

        x = x.masked_fill(mask.unsqueeze(-1), 0)  # (batch_size * inst_size, seq_size, dim)
        x = x.view(batch_size, inst_size, seq_size, -1)  # (batch_size, inst_size, seq_size, dim)
        return x

class DeepPMBasicBlock(nn.Module):
    def __init__(self, dim, dim_ff, n_heads, n_layers, dropout=None,
                 use_layernorm=False, layer_norm_eps=1e-05, use_checkpoint=False, activation='gelu', dummy=None,
                 handle_neg=False):
        super().__init__()

        self.tr = nn.ModuleList([DeepPMTransformerEncoderLayer(dim, n_heads, dim_ff,
                                 use_layernorm=use_layernorm, layer_norm_eps=layer_norm_eps,
                                 dropout=dropout, activation=activation, handle_neg=handle_neg)  for _ in range(n_layers)])

        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            if dummy is None:
                self.dummy = torch.zeros(1, requires_grad=True, device='cuda')
            else:
                self.dummy = dummy

    def forward(self, x, mask, weighted_attn=None):
        """
        x: [batch_size, inst_size, seq_size, dim]
        mask: [batch_size, inst_size, seq_size]
        """
        batch_size, inst_size, seq_size, _ = x.shape  # (batch_size, inst_size, seq_size, dim)

        x = x.view(batch_size, inst_size * seq_size, -1)  # (batch_size, inst_size * seq_size, dim)
        mask = mask.view(batch_size, inst_size * seq_size)  # (batch_size, inst_size * seq_size)

        for block in self.tr:
            if self.use_checkpoint:
                x = checkpoint(method_dummy_wrapper(block), self.dummy, x, mask, weighted_attn)
            else:
                x = block(x, mask, weighted_attn)

        x = x.masked_fill(mask.unsqueeze(-1), 0)  # (batch_size, inst_size * seq_size, dim)
        x = x.view(batch_size, inst_size, seq_size, -1)  # (batch_size, inst_size, seq_size, dim)
        return x

class DeepPMOp(nn.Module):
    def __init__(self, dim, dim_ff, n_heads, n_layers, dropout=None,
                 use_layernorm=False, layer_norm_eps=1e-05, use_checkpoint=False, activation='gelu', dummy=None,
                 handle_neg=False):
        super().__init__()

        self.tr = nn.ModuleList([DeepPMTransformerEncoderLayer(dim, n_heads, dim_ff,
                                 use_layernorm=use_layernorm, layer_norm_eps=layer_norm_eps,
                                 dropout=dropout, activation=activation, handle_neg=handle_neg)  for _ in range(n_layers)])

        self.use_checkpoint = use_checkpoint
        if self.use_checkpoint:
            if dummy is None:
                self.dummy = torch.zeros(1, requires_grad=True, device='cuda')
            else:
                self.dummy = dummy

    def forward(self, x, op_seq_mask, weighted_attn=None):
        """
        x: [batch_size, inst_size, seq_size, dim]
        op_seq_mask: [batch_size, inst_size]
        """
        for block in self.tr:
            if self.use_checkpoint:
                x = checkpoint(method_dummy_wrapper(block), self.dummy, x, op_seq_mask, weighted_attn)
            else:
                x = block(x, op_seq_mask, weighted_attn)

        x = x.masked_fill(op_seq_mask.unsqueeze(-1), 0)
        return x

class DeepPMTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, n_heads, dim_ff=2048,
                 use_layernorm=False,
                 layer_norm_eps=1e-05,
                 dropout=None,
                 activation='gelu',
                 handle_neg=False):
        super().__init__()

        if activation == 'gelu':
            act = nn.GELU
        elif activation == 'relu':
            act = nn.ReLU
        else:
            raise NotImplementedError()

        if dropout is None:
            dropout = 0.0

        self.attn = CustomSelfAttention(dim, n_heads, dropout, handle_neg=handle_neg)

        self.dropout = nn.Dropout(dropout)

        self.pwff = nn.Sequential(
            nn.Linear(dim, dim_ff),
            act(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim),
            nn.Dropout(dropout)
        )

        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)

    def forward(self, src, src_key_padding_mask=None, weighted_attn=None):
        x = src

        h = self.attn(x, key_padding_mask=src_key_padding_mask, attn_mask_modifier=weighted_attn)
        h = self.dropout(h)

        if self.use_layernorm:
            h = self.norm1(x + h)
            forward = self.pwff(h)
            h = self.norm2(h + forward)
        else:
            h = x + h
            forward = self.pwff(h)
            h = h + forward

        return h

class CustomSelfAttention(nn.Module):
    """
    CustomSelfAttention(dim, n_heads, dropout, handle_neg=handle_neg)
    """

    def __init__(self, dim, n_heads, dropout=None, handle_neg=False):
        super().__init__()

        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        if dropout is None:
            dropout = 0.0
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(dim, dim)

        self.handle_neg = handle_neg

    def forward(self, x, key_padding_mask=None, attn_mask_modifier=None):
        # B S D
        batch_size, seq_size, _ = x.shape

        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)

        # B S H W -trans-> B H S W
        q = q.view(batch_size, seq_size, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_size, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_size, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # B H S S
        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)

        if attn_mask_modifier is not None:
            if self.handle_neg:
                energy = energy - abs(energy * (1 - attn_mask_modifier.unsqueeze(1)))
            else:
                energy = energy * attn_mask_modifier.unsqueeze(1)

        if key_padding_mask is not None:
            energy = energy.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(-1), -1e10)

        attention = F.softmax(energy, dim=-1)

        # B H S W -> B S H W
        x = torch.matmul(self.dropout(attention), v).permute(0, 2, 1, 3).contiguous()

        # B S D
        x = x.view(batch_size, seq_size, -1)

        return self.output(x)