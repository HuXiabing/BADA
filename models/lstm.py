import torch
import torch.nn as nn

def get_last_non_padding_values(x, mask, dim):

    broadcast_shape = list(x.shape)
    broadcast_shape[dim] = 1

    indices = torch.argmax(mask.to(torch.int), dim=dim, keepdim=True)
    indices = indices.masked_fill(indices == 0, mask.size(dim))
    indices = indices - 1

    br = torch.broadcast_to(indices.unsqueeze(-1), broadcast_shape)
    output = torch.gather(x, dim, br)
    return output.squeeze(dim)


class BatchRNN(nn.Module):

    def __init__(self, embedding_size=512,
                 hidden_size=512,
                 pad_idx=0,
                 vocab_size=256):
        super(BatchRNN, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.pad_idx = pad_idx

        self.token_rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.instr_rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # output layer
        self.linear = nn.Linear(hidden_size, 1)
        self.set_learnable_embedding(vocab_size)

        self.to(self.device)

    def set_learnable_embedding(self, dictsize: int) -> None:

        embedding = nn.Embedding(dictsize, self.embedding_size)
        initrange = 0.5 / self.embedding_size
        embedding.weight.data.uniform_(-initrange, initrange)
        self.final_embeddings = embedding

    def forward(self, x):
        """

        Args:
            x: [batch_size, num_instructions, max_instruction_length]

        Returns:
            output: [batch_size]
        """
        mask = (x == self.pad_idx)
        batch_size, num_instr, seq_len = x.shape

        # 1. Token embedding
        # [batch_size, num_instructions, max_instruction_length, embedding_size]
        tokens = self.final_embeddings(x)

        # 2. Token-level RNN
        # [batch_size * num_instructions, max_instruction_length, embedding_size]
        tokens_reshaped = tokens.view(batch_size * num_instr, seq_len, -1)

        token_output, _ = self.token_rnn(tokens_reshaped)

        # [batch_size, num_instructions, max_instruction_length, hidden_size]
        token_output = token_output.view(batch_size, num_instr, seq_len, -1)

        # [batch_size, num_instructions, hidden_size]
        instr_representations = get_last_non_padding_values(token_output, mask, dim=2)

        # 3. Instruction-level RNN
        # [batch_size, num_instructions, hidden_size]
        instr_output, _ = self.instr_rnn(instr_representations)

        # 4. fical represent
        instr_mask = mask.all(dim=-1)

        # [batch_size, hidden_size]
        final_representation = get_last_non_padding_values(instr_output, instr_mask, dim=1)

        # 5. output
        output = self.linear(final_representation).squeeze(-1) # [batch_size]

        return output


class Fasthemal(nn.Module):

    def __init__(self, config):
        super(Fasthemal, self).__init__()

        self.config = config
        self.device = torch.device(config.device)

        self.model = BatchRNN(
            embedding_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            pad_idx=0,
            vocab_size=config.vocab_size
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, max_instr_count, max_instr_length]
        Returns:
            y: [batch_size]
        """
        return self.model(x)

    def count_parameters(self) -> int:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # print(f"Total trainable parameters: {total_params:,}")
        return total_params
