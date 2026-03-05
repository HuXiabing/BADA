TRANSFORMER_CONFIG = {
    "model_type": "transformer",
    "embed_dim": 128,
    "hidden_dim": 256,
    "num_layers": 1,
    "num_heads": 2,
    "dropout": 0.05,
    "use_layernorm": False,
    "use_checkpoint": False,
    "use_bb_attn": True,
    "use_seq_attn": True,
    "use_op_attn": True,
    "use_pos_2d": False,
    "handle_neg": False
}

ITHEMAL_CONFIG = {
    "model_type": "ithemal",
    "embed_dim": 128,
    "hidden_dim": 256,
    "num_layers": 1,
    "dropout": 0.1,
}

GNN_CONFIG = {
    "model_type": "gnn",
    "node_embedding_dim":64,
    "hidden_dim": 256,
    "message_passing_layers": 2,  # message passing layers
    "dropout": 0.1,
    "edge_embedding_dim": 32,
    "global_embedding_dim": 16,
    "use_layer_norm": True
}

# default data configuration
DATA_CONFIG = {
    "max_instr_length": 8,
    "max_instr_count": 64,
    "vocab_size": 256,
}

# default training configuration
TRAINING_CONFIG = {
    "lr": 2e-5,
    "weight_decay": 1e-5,
    "batch_size": 8,
    "epochs": 30,
    "patience": 5,
    "clip_grad_norm": 1.0,
    "loss_type": "mape",  # 'mse', 'mae', 'huber', 'mape'
    "loss_epsilon": 1e-5,  # for MAPE
    "huber_delta": 1.0,    # for Huber
    "optimizer": "adamw",  # 'adam', 'adamw'
    "scheduler": "cosine_warmup",  # 'plateau', 'cosine', 'cosine_warmup', 'step'
}

# merge all configurations
DEFAULT_CONFIG = {
    **DATA_CONFIG,
    **TRAINING_CONFIG,
    "output_dir": "experiments",
    "experiment_name": "test",
    "seed": 42,
    "verbose": True,
    "save_best_only": True,
    "save_freq": 1,
    "max_checkpoints": 3,
    "num_workers": 6
}
