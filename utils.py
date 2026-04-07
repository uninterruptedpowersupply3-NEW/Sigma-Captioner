import os
import json
import glob
from huggingface_hub import hf_hub_download, snapshot_download
from PyQt6.QtCore import QThread, pyqtSignal
from PIL import Image
from typing import Optional


def get_model_roots(config) -> list[str]:
    """
    Return a prioritized list of directories to search for model subfolders.

    Sigma-Captioner primarily uses `config["model_dir"]`, but we also include common
    Windows install locations (Program Files / LocalAppData\\Programs) to support
    users who moved the `models/` folder there.
    """
    model_dir = (config or {}).get("model_dir") or "./models"
    roots: list[str] = [model_dir]

    for env_key in ("PROGRAMFILES", "PROGRAMFILES(X86)"):
        program_files = os.environ.get(env_key)
        if not program_files:
            continue
        roots.append(os.path.join(program_files, "Sigma-Captioner", "models"))
        roots.append(os.path.join(program_files, "Sigma Captioner", "models"))

    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        # Common "per-user install" locations
        roots.append(os.path.join(local_appdata, "Programs", "Sigma-Captioner", "models"))
        roots.append(os.path.join(local_appdata, "Programs", "Sigma Captioner", "models"))
        roots.append(os.path.join(local_appdata, "Sigma-Captioner", "models"))
        roots.append(os.path.join(local_appdata, "Sigma Captioner", "models"))

    program_data = os.environ.get("PROGRAMDATA")
    if program_data:
        roots.append(os.path.join(program_data, "Sigma-Captioner", "models"))
        roots.append(os.path.join(program_data, "Sigma Captioner", "models"))

    # De-duplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for r in roots:
        r = os.path.normpath(r)
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out

class ModelDownloader(QThread):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(str)
    log = pyqtSignal(str)
    
    MODEL_REGISTRY = {
        # BLIP needs config + tokenizer files; otherwise Processor/Tokenizer load fails even if weights exist.
        "BLIP_CAP": {
            "repo_id": "Salesforce/blip-image-captioning-large",
            "presence_all": [
                "config.json",
                "preprocessor_config.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "vocab.txt",
            ],
            "presence_any": [
                "*.safetensors",
                "pytorch_model.bin",
            ],
        },
        "BLIP_VQA": {
            "repo_id": "Salesforce/blip-vqa-base",
            "presence_all": [
                "config.json",
                "preprocessor_config.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "vocab.txt",
            ],
            "presence_any": [
                "*.safetensors",
                "pytorch_model.bin",
            ],
        },
        "Florence-2": {"repo_id": "microsoft/Florence-2-large-ft"},
        "CLIP_HEAVY": {"repo_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"},
        "CLIP_LIGHT": {"repo_id": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"},
        "JoyCaption": {"repo_id": "mradermacher/llama-joycaption-beta-one-hf-llava-i1-GGUF", "filenames": ["llama-joycaption-beta-one-hf-llava-i1.Q4_K_M.gguf"]},
        "GIT": {"repo_id": "microsoft/git-large-textvqa"},
        "WD_Tagger": {"repo_id": "SmilingWolf/wd-eva02-large-tagger-v3"},
        "Moondream": {"repo_id": "vikhyatk/moondream2"},
        "SmolVLM": {"repo_id": "HuggingFaceTB/SmolVLM-256M-Instruct"}, # Using 1.7B as a powerful small choice
        # Qwen weights may be a single file or multiple shards; accept any *.safetensors.
        "Qwen": {
            "repo_id": "Qwen/Qwen3-VL-Embedding-2B",
            "presence_all": ["config.json", "preprocessor_config.json"],
            "presence_any": ["*.safetensors"],
        },
    }
    
    JSON_KEY_MAP = {
        "BLIP_CAP": "blip", "BLIP_VQA": "blip", "Florence-2": "florence",
        "CLIP_HEAVY": "clip_interrogator", "CLIP_LIGHT": "clip_interrogator",
        "JoyCaption": "llava", "GIT": "git", "WD_Tagger": "wd_tagger",
        "Moondream": "moondream", "SmolVLM": "smolvlm", "Qwen": "qwen",
    }

    @staticmethod
    def is_model_present(model_key: str, path: str, details: dict) -> bool:
        if not os.path.isdir(path):
            return False

        def _has_any(patterns: list[str]) -> bool:
            for pat in patterns:
                pat = (pat or "").strip()
                if not pat:
                    continue
                if any(ch in pat for ch in "*?[]"):
                    if glob.glob(os.path.join(path, pat)):
                        return True
                else:
                    if os.path.exists(os.path.join(path, pat)):
                        return True
            return False

        def _has_all(patterns: list[str]) -> bool:
            for pat in patterns:
                pat = (pat or "").strip()
                if not pat:
                    continue
                if any(ch in pat for ch in "*?[]"):
                    if not glob.glob(os.path.join(path, pat)):
                        return False
                else:
                    if not os.path.exists(os.path.join(path, pat)):
                        return False
            return True

        presence_all = details.get("presence_all") or details.get("presence_files")
        if presence_all and not _has_all(list(presence_all)):
            return False

        presence_any = details.get("presence_any")
        if presence_any and not _has_any(list(presence_any)):
            return False

        filenames = details.get("filenames")
        if filenames:
            return all(os.path.exists(os.path.join(path, f)) for f in filenames)

        # If no explicit presence rules are provided, require at least one weight-like file.
        # This prevents "config-only" partial snapshots from being treated as valid models.
        weight_candidates = []
        for pat in ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf"):
            weight_candidates.extend(glob.glob(os.path.join(path, pat)))
        if not weight_candidates:
            return False

        try:
            # Snapshot downloads should place at least one file in the directory.
            return any(entry.is_file() for entry in os.scandir(path))
        except OSError:
            return False

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
                 is_present = self.is_model_present(model_key, path, details)
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

                # Download only the minimal set of required file types (avoid ONNX exports, media, etc.)
                allow_patterns_base = details.get("allow_patterns") or [
                    "*.json",
                    "*.txt",
                    "*.model",
                    "*.py",
                    "*.csv",
                    "*.tiktoken",
                ]
                # Prefer safetensors first to avoid downloading both `.safetensors` and `pytorch_model.bin`.
                allow_patterns_stage1 = allow_patterns_base + ["*.safetensors"]
                allow_patterns_stage2 = allow_patterns_base + ["*.safetensors", "*.bin"]
                ignore_patterns = details.get("ignore_patterns") or [
                    "*.onnx",
                    "*.onnx_data",
                    "*.png",
                    "*.jpg",
                    "*.jpeg",
                    "*.gif",
                    "*.webp",
                    "*.mp4",
                    "*.mov",
                    "*.pdf",
                ]
                
                def _snapshot_download_with_retry(*, allow_patterns: list[str] | None):
                    """
                    snapshot_download can occasionally error with `httpx` client lifecycle issues
                    on flaky connections (e.g. WinError 10054). Retry once with `max_workers=1`
                    to reduce parallel HTTP pressure.
                    """
                    try:
                        snapshot_download(
                            repo_id=repo_id,
                            local_dir=local_path,
                            allow_patterns=allow_patterns,
                            ignore_patterns=ignore_patterns,
                            max_workers=8,
                        )
                    except RuntimeError as e:
                        if "client has been closed" in str(e).lower():
                            self.log.emit(f"Warning: download client closed for {model_key}; retrying with max_workers=1...")
                            snapshot_download(
                                repo_id=repo_id,
                                local_dir=local_path,
                                allow_patterns=allow_patterns,
                                ignore_patterns=ignore_patterns,
                                max_workers=1,
                            )
                        else:
                            raise

                if filenames:
                    for filename in filenames:
                        if not self._is_running: break
                        hf_hub_download(
                            repo_id=repo_id, filename=filename, local_dir=local_path,
                        )
                else:
                    try:
                        _snapshot_download_with_retry(allow_patterns=allow_patterns_stage1)
                        # If no safetensors were downloaded, fall back to `.bin` weights.
                        if self._is_running and not glob.glob(os.path.join(local_path, "*.safetensors")):
                            _snapshot_download_with_retry(allow_patterns=allow_patterns_stage2)
                    except TypeError:
                        # Back-compat for older huggingface_hub versions without allow_patterns.
                        snapshot_download(
                            repo_id=repo_id,
                            local_dir=local_path,
                            ignore_patterns=ignore_patterns,
                        )

                if self._is_running:
                    if self.is_model_present(model_key, local_path, details):
                        self.log.emit(f"Successfully downloaded {model_key} (Safetensors).")
                    else:
                        self.log.emit(
                            f"ERROR: {model_key} download finished but required files are still missing in: {local_path}"
                        )
            except Exception as e:
                self.log.emit(f"Error downloading {model_key}: {e}")
        self.finished.emit("Downloads completed." if self._is_running else "Download stopped.")


def resolve_model_dir(model_key: str, config: dict, details: Optional[dict] = None) -> Optional[str]:
    """
    Try to resolve a usable local directory for `model_key`.

    Search order:
    1) `config["model_dir"]` (portable) + common Program Files install roots
    2) Hugging Face Hub cache (local_files_only)

    Returns a directory path if found, otherwise None.
    """
    if details is None:
        details = ModelDownloader.MODEL_REGISTRY.get(model_key)

    roots = get_model_roots(config)

    if details:
        for root in roots:
            candidate = os.path.join(root, model_key)
            if ModelDownloader.is_model_present(model_key, candidate, details):
                return candidate

        repo_id = (details.get("repo_id") or "").strip()
        if repo_id:
            try:
                cached = snapshot_download(repo_id, local_files_only=True)
            except Exception:
                cached = None
            if cached and ModelDownloader.is_model_present(model_key, cached, details):
                return cached
        return None

    for root in roots:
        candidate = os.path.join(root, model_key)
        if os.path.isdir(candidate):
            return candidate
    return None

def get_model_path(model_key, config):
    model_dir = config.get('model_dir', './models')
    if model_key == "CLIP_Interrogator":
        variant = config.get('clip_model_variant', 'light')
        key = 'CLIP_LIGHT' if variant == 'light' else 'CLIP_HEAVY'
        resolved = resolve_model_dir(key, config)
        return resolved if resolved else os.path.join(model_dir, key)
    if model_key == "BLIP":
        resolved = resolve_model_dir('BLIP_CAP', config)
        return resolved if resolved else os.path.join(model_dir, 'BLIP_CAP')
    resolved = resolve_model_dir(model_key, config)
    return resolved if resolved else os.path.join(model_dir, model_key)

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
            'git': True, 'wd_tagger': True, 'moondream': True, 'smolvlm': True, 'qwen': False, 'sglang': False
        },
        'models_vqa_enabled': {
            'blip': True, 'florence': True, 'llava': False, 'git': True,
            'moondream': True, 'smolvlm': True,
        },
        'model_specific_batch_sizes': {
            'clip_interrogator': 1, 'blip': 8, 'florence': 4, 'llava': 1, 
            'git': 4, 'wd_tagger': 8, 'moondream': 8, 'smolvlm': 8, 'qwen': 8, 'sglang': 8
        },
        'model_specific_max_words': {
            'blip': 75, 'florence': 1024, 'llava': 256, 'git': 100,
            'moondream': 75, 'smolvlm': 256, 'qwen': 50, 'sglang': 256
        },
        'model_specific_parameters': {
            'smolvlm_qa_pairs': 3 # Number of Question/Answer pairs to generate
        },
         'qwen_precision': 'bf16', 'qwen_output_dim': 512, 'qwen_tf32': True, 'qwen_quant': False,
        'qwen_compile': True, 'qwen_dynamic': True, 'qwen_cuda_graphs': False, 'qwen_pinned_mem': True,
        'qwen_threshold': 0.30, 'qwen_max_tags': 50, 'qwen_legacy_support': False,
        'qwen_use_json_cache': True, 'qwen_json_cache_path': '', 'qwen_prefetch': 2,
        'qwen_inductor_cache_dir': '',
        'qwen_inductor_compile_threads': 0,
         'sglang_url': 'http://127.0.0.1:30000/generate',
          'sglang_use_native_generate': True,
          'sglang_shutdown_wsl_on_unload': True,
          'sglang_disable_reasoning': True,
          'sglang_format_reasoning': True,
          'sglang_system_context': "You are a precise, data-extraction expert. Analyze the image and generate a technical test bank question.\nRULES:\n1. Output STRICTLY in raw JSON format.\n2. NO markdown wrappers.\n3. You MUST place all your internal reasoning inside the 'thought_process' JSON key.",
          'sglang_concurrency': 40, 'sglang_max_tokens': 1024, 'sglang_max_res': 256, 'sglang_legacy_support': False,
          'sglang_auto_wsl': False,
          'sglang_wsl_activate': '',
          'sglang_wsl_model_path': '',
          'sglang_wsl_launch': '',
          'sglang_wsl_extra_args': '',
          'sglang_wsl_cmd': ''
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
