from pathlib import Path
from typing import Dict, Any
import yaml
import os
import torch
import logging

# module-level logger
logger = logging.getLogger(__name__)


try:
    from huggingface_hub import hf_hub_download
    import huggingface_hub
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

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

def load_model_asset(source: str, handle: str, filename: str = None) -> str:
    """
    Tải file model từ nguồn được chỉ định (kaggle, huggingface, local).
    Trả về đường dẫn cục bộ đến file/thư mục đã tải.
    """
    source = source.lower()
    if source == 'kaggle':
        return load_from_kaggle(handle, filename)
    elif source == 'huggingface':
        return load_from_huggingface(handle, filename)
    elif source == 'local':
        if not os.path.exists(handle):
            raise FileNotFoundError(f"Không tìm thấy file/thư mục local: {handle}")
        logger.info(f"Sử dụng đường dẫn local: {handle}")
        if filename and os.path.isdir(handle):
             full_path = os.path.join(handle, filename)
             if not os.path.exists(full_path):
                 raise FileNotFoundError(f"Không tìm thấy file '{filename}' trong thư mục local: {handle}")
             return full_path
        return handle 
    else:
        raise ValueError(f"Nguồn '{source}' không được hỗ trợ. Chỉ hỗ trợ 'kaggle', 'huggingface', 'local'.")

def load_from_huggingface(model_handle: str, filename: str = None) -> str:
    """
    Tải model hoặc file cụ thể từ Hugging Face Hub.
    Trả về đường dẫn cục bộ đến file hoặc thư mục đã tải.
    """
    if not HUGGINGFACE_AVAILABLE:
        raise RuntimeError("Thư viện 'huggingface_hub' chưa được cài đặt.")
        
    logger.info(f"Đang tải từ Hugging Face Hub: {model_handle} (file: {filename or 'toàn bộ repo'})")
    try:
        if filename:
            downloaded_path = hf_hub_download(repo_id=model_handle, filename=filename)
            logger.info(f"Đã tải file '{filename}' về: {downloaded_path}")
            return downloaded_path
        else:
             raise ValueError("Cần chỉ định 'filename' khi tải checkpoint tùy chỉnh (.pt) từ Hugging Face Hub.")
    except Exception as e:
        logger.error(f"Lỗi khi tải từ Hugging Face Hub {model_handle}: {e}")
        raise

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
