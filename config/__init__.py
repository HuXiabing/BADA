from .config import Config
from .defaults import (
    DEFAULT_CONFIG, 
    TRANSFORMER_CONFIG, 
    GNN_CONFIG,
    ITHEMAL_CONFIG
)

def get_config(model_type="transformer", **kwargs):

    config_dict = DEFAULT_CONFIG.copy()
    if model_type == "transformer":
        config_dict.update(TRANSFORMER_CONFIG)
    elif model_type == "lstm":
        config_dict.update(ITHEMAL_CONFIG)
    elif model_type == "gnn":
        config_dict.update(GNN_CONFIG)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    config_dict.update(kwargs)
    config_dict["model_type"] = model_type

    return Config(**config_dict)

