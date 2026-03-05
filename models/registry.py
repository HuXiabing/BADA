from typing import Dict, Type, Optional
import torch.nn as nn
from .transformer import TransformerModel
from .lstm import Fasthemal
from .gnn import RISCVGraniteModel

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "transformer": TransformerModel,
    "gnn": RISCVGraniteModel,
    "lstm": Fasthemal
}

def register_model(name: str, model_class: Type[nn.Module]) -> None:
    """
    Register a new model to the registry

    Args:
        name: Model name
        model_class: Model class
    """
    global MODEL_REGISTRY
    if name in MODEL_REGISTRY:
        raise ValueError(f"Model {name} is already registered")

    MODEL_REGISTRY[name] = model_class
    print(f"Model {name} registered")

def get_model(config, model_type: Optional[str] = None) -> nn.Module:
    """
    Create a model instance based on the configuration

    Args:
        config: Configuration object
        model_type: Model type, if None, use model_type from config

    Returns:
        Model instance
    """
    model_type = model_type or config.model_type
    
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_type}")
    
    model_class = MODEL_REGISTRY[model_type]
    model = model_class(config)
    
    return model

def list_available_models() -> Dict[str, Type[nn.Module]]:
    """
    List all available models
    Returns:
        Model registry
    """
    return MODEL_REGISTRY.copy()
