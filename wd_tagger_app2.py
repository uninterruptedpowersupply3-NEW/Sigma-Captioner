import torch
import gc
import os
import random
from PIL import Image
import numpy as np
import pandas as pd
import timm
import torchvision.transforms.functional as TF
from models import BaseModelWrapper
import platform

# --- Constants ---
DEFAULT_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"

class WDTaggerApp2Model(BaseModelWrapper):
    def __init__(self, model_key, model_path, config):
        super().__init__(model_key, model_path, config)
        self.labels = None
        self.input_size = (448, 448)
        self.repo_id = DEFAULT_REPO
        # --- NEW STATE ATTRIBUTES for JIT Compilation ---
        self.is_compiled = False
        self.current_compile_mode = 'none'
        self.had_graph_error = False

    def _load_model_specific(self):
        self._log("Loading WDv3 Tagger model (timm)...")
        # Ensure labels are loaded before model creation
        num_classes = len(self._load_labels()['names'])
        self.model = timm.create_model(f"hf-hub:{self.repo_id}", pretrained=True, num_classes=num_classes).eval().to(self.config.get("device", "cuda"))
        
        try:
            from safetensors.torch import load_file as load_safetensors
            state_dict_path = os.path.join(self.model_path, "model.safetensors")
            if os.path.exists(state_dict_path):
                self._log(f"Loading local state from {state_dict_path}")
                state_dict = load_safetensors(state_dict_path)
                self.model.load_state_dict(state_dict)
                self._log("Successfully loaded local safetensors.")
        except Exception as e:
            self._log(f"Could not load local safetensors, relying on pretrained weights from timm. Error: {e}")

        cfg = getattr(self.model, "pretrained_cfg", {}) or getattr(self.model, "default_cfg", {})
        if input_size := cfg.get("input_size"):
            self.input_size = (input_size[1], input_size[2])
        self._log(f"Model input size set to: {self.input_size}")

    def _load_labels(self):
        if self.labels: return self.labels
        self._log("Loading WDv3 tags...")
        csv_path = os.path.join(self.model_path, "selected_tags.csv")
        df = pd.read_csv(csv_path, usecols=["name", "category"])
        self.labels = {
            'names': df["name"].tolist(),
            'rating': list(np.where(df["category"] == 9)[0]),
            'general': list(np.where(df["category"] == 0)[0]),
            'character': list(np.where(df["category"] == 4)[0])
        }
        return self.labels

    def unload(self):
        super().unload()
        del self.labels
        self.labels = None
        self.is_compiled = False
        self.current_compile_mode = 'none'
        self.had_graph_error = False

    def compile_model(self, images_for_check=None):
        if self.is_compiled and self.current_compile_mode != 'none':
            return

        use_graphs = self.config.get('use_cuda_graphs', False)
        target_mode = 'fullgraph' if use_graphs and not self.had_graph_error else 'reduce-overhead'

        # On Windows, force backend to 'eager' to avoid Triton dependency
        backend = 'inductor'
        if platform.system() == 'Windows':
            backend = 'eager'
            self._log("Windows detected: using torch.compile with backend='eager' (no Triton)")

        print(f"\n--- Compiling WD Tagger in '{target_mode.upper()}' mode using backend='{backend}' ---")
        try:
            self.model = torch.compile(
                self.model,
                mode="reduce-overhead",
                fullgraph=(target_mode == 'fullgraph'),
                backend=backend
            )
            self.current_compile_mode = target_mode
            self._log("WD Tagger compilation successful.")
        except Exception as e:
            self._log(f"WD Tagger COMPILATION FAILED: {e}")
            self.current_compile_mode = 'none'
            if target_mode == 'fullgraph':
                self.had_graph_error = True
            # Optional: continue without crashing
            return
        self.is_compiled = True
        print("------------------------------------------------------------------\n")


    @torch.no_grad()
    def _infer_model_specific(self, images, **kwargs):
        # Trigger the one-time, optimistic compilation on the first batch
        if not self.is_compiled:
            self.compile_model()

        self._log(f"Processing batch of {len(images)} images with WDv3 Tagger...")
        batch = self._preprocess(images).to(self.config.get("device", "cuda"))

        logits = self.model(batch)
        probs_cpu = torch.sigmoid(logits).cpu()
        
        gen_th = self.config.get('waifu_diffusion_general_threshold', 0.35)
        char_th = self.config.get('waifu_diffusion_character_threshold', 0.85)

        general_indices = torch.tensor(self.labels['general'], dtype=torch.long)
        char_indices = torch.tensor(self.labels['character'], dtype=torch.long)
        
        batch_results = []
        for i in range(probs_cpu.size(0)):
            gen_probs = probs_cpu[i, general_indices]
            char_probs = probs_cpu[i, char_indices]

            gen_selected = [self.labels['names'][idx] for prob, idx in zip(gen_probs, self.labels['general']) if prob > gen_th]
            char_selected = [self.labels['names'][idx] for prob, idx in zip(char_probs, self.labels['character']) if prob > char_th]
            
            parts = []
            for key in ("1girl", "1boy", "1other"):
                if key in gen_selected:
                    parts.append(key)
                    gen_selected.remove(key)
                    break
            parts.extend(char_selected)
            series = [tag for tag in gen_selected if "from" in tag.lower()]
            for s in series:
                parts.append(s)
                gen_selected.remove(s)
            
            caption = ", ".join(parts + gen_selected).replace("_", " ")
            batch_results.append({"caption": caption})
            
        return batch_results

    def _preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.config.get("device", "cuda")).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.config.get("device", "cuda")).view(3, 1, 1)
        
        tensors = []
        for img in images:
            img_rgb = img.convert("RGB")
            tensor = TF.to_tensor(img_rgb)
            if tensor.shape[1:] != self.input_size:
                tensor = TF.resize(tensor, list(self.input_size), antialias=True)
            tensors.append(tensor)
            
        batch_tensor = torch.stack(tensors).to(self.config.get("device", "cuda"))
        batch_tensor = (batch_tensor - mean) / std
        return batch_tensor