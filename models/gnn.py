import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from data.gnn_dataset import RISCVGraphEncoder

class RISCVGraniteModel(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(RISCVGraniteModel, self).__init__()

        graph_encoder = RISCVGraphEncoder()
        self.vocab_size = graph_encoder.get_vocab_size()
        self.num_edge_types = graph_encoder.get_num_edge_types()

        self.node_embedding_dim = config.node_embedding_dim
        self.edge_embedding_dim = config.edge_embedding_dim
        self.global_embedding_dim = config.global_embedding_dim
        self.hidden_dim = config.hidden_dim
        self.num_message_passing_steps = config.message_passing_layers
        self.dropout = config.dropout
        self.use_layer_norm = config.use_layer_norm

        self.gnn = GraphNeuralNetwork(
            node_embedding_dim=self.node_embedding_dim,
            edge_embedding_dim=self.edge_embedding_dim,
            global_embedding_dim=self.global_embedding_dim,
            hidden_dim=self.hidden_dim,
            num_message_passing_steps=self.num_message_passing_steps,
            dropout=self.dropout,
            use_layer_norm=self.use_layer_norm,
            vocab_size=self.vocab_size,
            num_edge_types=self.num_edge_types
        )

        self.decoder = ThroughputDecoder(
            node_embedding_dim=self.node_embedding_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            use_layer_norm=self.use_layer_norm,
        )

    def count_parameters(self) -> int:
        # print("=" * 60)
        # print(" GRANITE Model Parameters:")
        # print(f"  node_embedding_dim: {self.node_embedding_dim}")
        # print(f"  edge_embedding_dim: {self.edge_embedding_dim}")
        # print(f"  global_embedding_dim: {self.global_embedding_dim}")
        # print(f"  hidden_dim: {self.hidden_dim}")
        # print(f"  num_message_passing_steps: {self.num_message_passing_steps}")
        # print(f"  dropout: {self.dropout}")
        # print(f"  use_layer_norm: {self.use_layer_norm}")
        # print(f"  vocab_size: {self.vocab_size}")
        # print(f"  num_edge_types: {self.num_edge_types}")

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return total_params

    def forward(self, basic_block_graph):

        node_embeddings, _, _ = self.gnn(
            basic_block_graph.x,
            basic_block_graph.edge_index,
            basic_block_graph.edge_attr,
            basic_block_graph.batch if hasattr(basic_block_graph, 'batch') else None
        )

        instruction_mask = basic_block_graph.instruction_mask
        instruction_embeddings = node_embeddings[instruction_mask]

        if hasattr(basic_block_graph, 'batch'):
            instruction_batch = basic_block_graph.batch[instruction_mask]
        else:
            instruction_batch = None

        output = self.decoder(instruction_embeddings, instruction_batch)

        return output

class GraphNeuralNetwork(nn.Module):

    def __init__(
            self,
            node_embedding_dim: int = 256,
            edge_embedding_dim: int = 256,
            global_embedding_dim: int = 256,
            hidden_dim: int = 256,
            num_message_passing_steps: int = 2,
            dropout: float = 0.1,
            use_layer_norm: bool = True,
            vocab_size: int = 300,
            num_edge_types: int = 10,
    ):

        super(GraphNeuralNetwork, self).__init__()

        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.global_embedding_dim = global_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_message_passing_steps = num_message_passing_steps
        self.use_layer_norm = use_layer_norm
        self.vocab_size = vocab_size
        self.num_edge_types = num_edge_types

        self.node_embedding = nn.Embedding(vocab_size, node_embedding_dim)
        self.edge_embedding = nn.Embedding(num_edge_types, edge_embedding_dim)
        self.global_init = nn.Linear(vocab_size + num_edge_types, global_embedding_dim)

        self.message_passing_layers = nn.ModuleList([
            MessagePassingLayer(
                node_embedding_dim=node_embedding_dim,
                edge_embedding_dim=edge_embedding_dim,
                global_embedding_dim=global_embedding_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_layer_norm=use_layer_norm
            ) for _ in range(num_message_passing_steps)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:

        node_embeddings = self.node_embedding(x)  # [num_nodes]
        edge_embeddings = self.edge_embedding(edge_attr.squeeze(-1))

        if batch is None:
            global_features = self.compute_global_features(x, edge_attr.squeeze(-1))
            global_embedding = self.global_init(global_features.unsqueeze(0))  # [1, global_dim]
        else:
            num_graphs = batch.max().item() + 1
            global_embeddings = []

            for i in range(num_graphs):

                node_mask = batch == i
                graph_nodes = x[node_mask]

                edge_mask = batch[edge_index[0]] == i
                graph_edges = edge_attr.squeeze(-1)[edge_mask]

                global_feat = self.compute_global_features(graph_nodes, graph_edges)
                global_embeddings.append(global_feat)

            global_features_batch = torch.stack(global_embeddings)  # [num_graphs, feature_dim]
            global_embedding = self.global_init(global_features_batch)

        for i in range(self.num_message_passing_steps):
            node_embeddings, edge_embeddings, global_embedding = self.message_passing_layers[i](
                node_embeddings, edge_embeddings, global_embedding, edge_index, batch
            )

        return node_embeddings, edge_embeddings, global_embedding

    def compute_global_features(self, node_tokens: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:

        node_counts = torch.bincount(node_tokens, minlength=self.vocab_size)
        node_freqs = node_counts.float() / node_tokens.size(0) if node_tokens.size(0) > 0 else node_counts.float()

        edge_counts = torch.bincount(edge_types, minlength=self.num_edge_types)
        edge_freqs = edge_counts.float() / edge_types.size(0) if edge_types.size(0) > 0 else edge_counts.float()

        global_features = torch.cat([node_freqs, edge_freqs])

        return global_features

class MessagePassingLayer(nn.Module):

    def __init__(
            self,
            node_embedding_dim: int,
            edge_embedding_dim: int,
            global_embedding_dim: int,
            hidden_dim: int,
            dropout: float = 0.1,
            use_layer_norm: bool = True,
    ):

        super(MessagePassingLayer, self).__init__()

        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.global_embedding_dim = global_embedding_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm

        self.edge_update = nn.Sequential(
            nn.Linear(edge_embedding_dim + 2 * node_embedding_dim + global_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_embedding_dim)
        )

        self.node_update = nn.Sequential(
            nn.Linear(node_embedding_dim + hidden_dim + global_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_embedding_dim)
        )

        self.global_update = nn.Sequential(
            nn.Linear(global_embedding_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, global_embedding_dim)
        )

        if use_layer_norm:
            self.edge_layer_norm = nn.LayerNorm(edge_embedding_dim)
            self.node_layer_norm = nn.LayerNorm(node_embedding_dim)
            self.global_layer_norm = nn.LayerNorm(global_embedding_dim)

        self.edge_to_message = nn.Linear(edge_embedding_dim, hidden_dim)
        self.node_to_global = nn.Linear(node_embedding_dim, hidden_dim)
        self.edge_to_global = nn.Linear(edge_embedding_dim, hidden_dim)

    def forward(self, node_embeddings: torch.Tensor, edge_embeddings: torch.Tensor,
                global_embedding: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        num_nodes = node_embeddings.size(0)
        num_edges = edge_index.size(1)

        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=node_embeddings.device)

        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        src_embeddings = node_embeddings[src_nodes]
        dst_embeddings = node_embeddings[dst_nodes]

        edge_global_embeddings = global_embedding[batch[src_nodes]]

        edge_inputs = torch.cat([
            src_embeddings,
            dst_embeddings,
            edge_embeddings,
            edge_global_embeddings
        ], dim=1)

        edge_updates = self.edge_update(edge_inputs)

        updated_edge_embeddings = edge_embeddings + edge_updates
        if self.use_layer_norm:
            updated_edge_embeddings = self.edge_layer_norm(updated_edge_embeddings)

        edge_messages = self.edge_to_message(updated_edge_embeddings)

        node_messages = torch.zeros(num_nodes, self.hidden_dim, device=node_embeddings.device)

        for i in range(num_edges):
            node_messages[dst_nodes[i]] += edge_messages[i]

        node_global_embeddings = global_embedding[batch]
        node_inputs = torch.cat([
            node_embeddings,
            node_messages,
            node_global_embeddings
        ], dim=1)

        node_updates = self.node_update(node_inputs)

        updated_node_embeddings = node_embeddings + node_updates
        if self.use_layer_norm:
            updated_node_embeddings = self.node_layer_norm(updated_node_embeddings)

        num_graphs = global_embedding.size(0)

        node_features_for_global = self.node_to_global(updated_node_embeddings)

        edge_features_for_global = self.edge_to_global(updated_edge_embeddings)

        node_aggregated = torch.zeros(num_graphs, self.hidden_dim, device=node_embeddings.device)
        for i in range(num_nodes):
            node_aggregated[batch[i]] += node_features_for_global[i]

        edge_aggregated = torch.zeros(num_graphs, self.hidden_dim, device=edge_embeddings.device)
        for i in range(num_edges):
            edge_aggregated[batch[src_nodes[i]]] += edge_features_for_global[i]

        global_inputs = torch.cat([
            global_embedding,
            node_aggregated,
            edge_aggregated
        ], dim=1)

        global_updates = self.global_update(global_inputs)

        updated_global_embedding = global_embedding + global_updates
        if self.use_layer_norm:
            updated_global_embedding = self.global_layer_norm(updated_global_embedding)

        return updated_node_embeddings, updated_edge_embeddings, updated_global_embedding

class ThroughputDecoder(nn.Module):

    def __init__(
            self,
            node_embedding_dim: int = 256,
            hidden_dim: int = 256,
            dropout: float = 0.1,
            use_layer_norm: bool = True,
    ):
        super(ThroughputDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(node_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(node_embedding_dim)

    def forward(self, instruction_embeddings, batch=None):

        if self.use_layer_norm:
            instruction_embeddings = self.layer_norm(instruction_embeddings)

        instruction_contributions = self.decoder(instruction_embeddings).squeeze(-1)

        if batch is None:
            return torch.sum(instruction_contributions).unsqueeze(0)  # [1]

        batch_size = batch.max().item() + 1
        throughputs = torch.zeros(batch_size, device=instruction_embeddings.device)

        throughputs.scatter_add_(0, batch, instruction_contributions)

        return throughputs  # [batch_size]