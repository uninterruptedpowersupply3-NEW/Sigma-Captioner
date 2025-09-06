import os
import json
from huggingface_hub import hf_hub_download, snapshot_download
from PyQt6.QtCore import QThread, pyqtSignal
from PIL import Image

class ModelDownloader(QThread):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(str)
    log = pyqtSignal(str)
    
    MODEL_REGISTRY = {
        "BLIP_CAP": {"repo_id": "Salesforce/blip-image-captioning-large"},
        "BLIP_VQA": {"repo_id": "Salesforce/blip-vqa-base"},
        "Florence-2": {"repo_id": "microsoft/Florence-2-large-ft"},
        "CLIP_HEAVY": {"repo_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"},
        "CLIP_LIGHT": {"repo_id": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"},
        "JoyCaption": {"repo_id": "mradermacher/llama-joycaption-beta-one-hf-llava-i1-GGUF", "filenames": ["llama-joycaption-beta-one-hf-llava-i1.Q4_K_M.gguf"]},
        "GIT": {"repo_id": "microsoft/git-large-textvqa"},
        "WD_Tagger": {"repo_id": "SmilingWolf/wd-eva02-large-tagger-v3"},
        "Moondream": {"repo_id": "vikhyatk/moondream2"},
        "SmolVLM": {"repo_id": "HuggingFaceTB/SmolVLM-256M-Instruct"}, # Using 1.7B as a powerful small choice
    }
    
    JSON_KEY_MAP = {
        "BLIP_CAP": "blip", "BLIP_VQA": "blip", "Florence-2": "florence",
        "CLIP_HEAVY": "clip_interrogator", "CLIP_LIGHT": "clip_interrogator",
        "JoyCaption": "llava", "GIT": "git", "WD_Tagger": "wd_tagger",
        "Moondream": "moondream", "SmolVLM": "smolvlm",
    }

    def __init__(self, model_dir, config):
        super().__init__()
        self.model_dir = model_dir
        self.config = config
        self._is_running = True

    def stop(self): self._is_running = False

    def run(self):
        models_to_download = []
        for model_key, json_key in self.JSON_KEY_MAP.items():
            if not self._is_running: break
            if "CLIP" in model_key:
                if self.config['clip_model_variant'] == 'heavy' and model_key != 'CLIP_HEAVY': continue
                if self.config['clip_model_variant'] == 'light' and model_key != 'CLIP_LIGHT': continue
            if self.config['models_enabled'].get(json_key, False):
                details = self.MODEL_REGISTRY[model_key]
                path = os.path.join(self.model_dir, model_key)
                is_present = os.path.isdir(path) and (not details.get("filenames") or all(os.path.exists(os.path.join(path, f)) for f in details["filenames"]))
                if not is_present: models_to_download.append(model_key)
        if not self._is_running: self.finished.emit("Download stopped."); return
        if not models_to_download: self.finished.emit("All enabled models are already present."); return
        for model_key in models_to_download:
            if not self._is_running: break
            try:
                details = self.MODEL_REGISTRY[model_key]
                repo_id = details['repo_id']
                filenames = details.get('filenames')
                local_path = os.path.join(self.model_dir, model_key)
                os.makedirs(local_path, exist_ok=True)
                
                self.log.emit(f"Downloading {model_key} from {repo_id}...")
                self.progress.emit(f"Downloading {model_key}...", 0)

                # --- START OF MODIFICATION ---
                ignore_patterns = ["*.onnx", "*.onnx_data"] # Patterns to exclude
                
                if filenames:
                    for filename in filenames:
                        if not self._is_running: break
                        hf_hub_download(
                            repo_id=repo_id, filename=filename, local_dir=local_path,
                            local_dir_use_symlinks=False, resume_download=True
                        )
                else:
                    snapshot_download(
                        repo_id=repo_id, local_dir=local_path,
                        local_dir_use_symlinks=False, resume_download=True,
                        ignore_patterns=ignore_patterns # <-- ADD THIS ARGUMENT
                    )
                # --- END OF MODIFICATION ---

                if self._is_running:
                    self.log.emit(f"Successfully downloaded {model_key} (Safetensors).")
            except Exception as e:
                self.log.emit(f"Error downloading {model_key}: {e}")
        self.finished.emit("Downloads completed." if self._is_running else "Download stopped.")

def get_model_path(model_key, config):
    model_dir = config.get('model_dir', './models')
    if model_key == "CLIP_Interrogator":
        variant = config.get('clip_model_variant', 'light')
        return os.path.join(model_dir, 'CLIP_LIGHT' if variant == 'light' else 'CLIP_HEAVY')
    if model_key == "BLIP":
        return os.path.join(model_dir, 'BLIP_CAP')
    return os.path.join(model_dir, model_key)

def save_config(config):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    default_config = {
        'model_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'),
        'image_dir': os.path.join(os.path.expanduser('~'), 'Pictures'),
        'output_dir': os.path.join(os.path.expanduser('~'), 'Pictures', 'output'),
        'use_system_image_limits': True, 'max_width': 2048, 'max_height': 2048,
        'use_cuda_graphs': False, 'concurrent_loading': False, 'resume_processing': True,
        'use_question_file': False, 'question_json_path': '',
        'common_question': 'What is the main subject of this image?',
        'waifu_diffusion_general_threshold': 0.35, 'waifu_diffusion_character_threshold': 0.85,
        'llava_n_gpu_layers': -1, 'llava_n_ctx': 4096,
        'clip_model_variant': 'light', 'florence_caption_style': 'Detailed',
        'florence_enable_vqa': True, 'florence_enable_od': True,
        'florence_enable_dense_caption': True, 'florence_enable_ocr': False,
        'florence_enable_ocr_with_region': False, 'florence_enable_region_proposal': False,
        'florence_enable_caption_grounding': False, 'florence_filter_ocr': True,
        'moondream_revision': '2024-05-20', 'moondream_enable_vqa': True,
        'use_moondream_question_file': False, 'moondream_question_json_path': '',
        'models_enabled': {
            'clip_interrogator': True, 'blip': True, 'florence': True, 'llava': False, 
            'git': True, 'wd_tagger': True, 'moondream': True, 'smolvlm': True,
        },
        'models_vqa_enabled': {
            'blip': True, 'florence': True, 'llava': False, 'git': True,
            'moondream': True, 'smolvlm': True,
        },
        'model_specific_batch_sizes': {
            'clip_interrogator': 1, 'blip': 8, 'florence': 4, 'llava': 1, 
            'git': 4, 'wd_tagger': 8, 'moondream': 8, 'smolvlm': 8,
        },
        'model_specific_max_words': {
            'blip': 75, 'florence': 1024, 'llava': 256, 'git': 100,
            'moondream': 75, 'smolvlm': 256,
        },
                'model_specific_parameters': {
            'smolvlm_qa_pairs': 3 # Number of Question/Answer pairs to generate
        }
    }
    if not os.path.exists(config_path):
        save_config(default_config)
        return default_config
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        for key, default_val in default_config.items():
            if key not in config: config[key] = default_val
            elif isinstance(default_val, dict):
                for sub_key, sub_default_val in default_val.items():
                    if sub_key not in config[key]: config[key][sub_key] = sub_default_val
    except (FileNotFoundError, json.JSONDecodeError):
        config = default_config
        save_config(config)
    return config