from pathlib import Path
from typing import Dict, Any
import yaml
import os
import torch


def load_config(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def get_model_entry(cfg: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    models = cfg.get('models', {})
    entry = models.get(model_name, {})
    return entry


def get_handle(cfg: Dict[str, Any], model_name: str) -> str:
    entry = get_model_entry(cfg, model_name)
    return entry.get('handle')


def get_source(cfg: Dict[str, Any], model_name: str) -> str:
    entry = get_model_entry(cfg, model_name)
    return entry.get('source', 'kaggle')


def load_from_kaggle(model_handle: str, filename: str = None) -> str:
    try:
        import kagglehub
    except Exception as e:
        raise RuntimeError("kagglehub is required for 'kaggle' source: " + str(e))

    model_dir = kagglehub.model_download(model_handle)
    if filename:
        candidate = os.path.join(model_dir, filename)
        if os.path.exists(candidate):
            return candidate
    pt_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.pt')]
    if pt_files:
        return pt_files[0]
    return model_dir


def load_from_huggingface(handle: str) -> str:
    return 


def load_checkpoint_to_model(model_instance: torch.nn.Module, checkpoint_path: str, device=None):
    device = device or torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    first_key = next(iter(state_dict.keys()))
    if first_key.startswith('module.'):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model_instance.load_state_dict(state_dict, strict=False)
    model_instance.to(device)
    model_instance.eval()
    return model_instance
