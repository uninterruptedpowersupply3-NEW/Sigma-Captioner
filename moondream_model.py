# moondream_model.py
# Final, stable implementation using transformers, with torch.compile disabled for compatibility.

import os
import torch
import gc
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from models import BaseModelWrapper
from smolvlm_model import SmolVLMModel 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

class MoondreamModel(BaseModelWrapper):
    def __init__(self, model_key, model_path, config):
        super().__init__(model_key, model_path, config)
        self.tokenizer = None
        self.attn_impl = "N/A"
        # No compilation state needed anymore
        
    def _load_model_specific(self):
        # We revert to the stable loading from the local path.
        # This assumes the model has been downloaded by the app's downloader.
        if not os.path.isdir(self.model_path) or not os.listdir(self.model_path):
            self._log("Moondream model directory is missing or empty. Please download the model first.")
            raise FileNotFoundError("Moondream model files not found in the local model directory.")

        self._log(f"Loading model from local path: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        common_kwargs = {"trust_remote_code": True}
        if DEVICE == "cuda":
            common_kwargs['torch_dtype'] = DTYPE

        try:
            self._log("Attempting to load with Flash Attention 2...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, attn_implementation="flash_attention_2", **common_kwargs
            ).to(DEVICE).eval()
            self.attn_impl = "Flash Attention 2"
        except Exception:
            self._log("Flash Attention 2 failed, falling back to Eager.")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, attn_implementation="eager", **common_kwargs
            ).to(DEVICE).eval()
            self.attn_impl = "Eager (Default)"
        
        print("\n--- Moondream Model Optimizations Loaded ---")
        print(f"  > Precision:                {DTYPE if DEVICE == 'cuda' else 'torch.float32'}")
        print(f"  > Attention Implementation: {self.attn_impl}")
        print(f"  > JIT Compilation:          NO (disabled for compatibility)")
        print("-------------------------------------------\n")

    def unload(self):
        if hasattr(self, 'model') and self.model is not None: del self.model; self.model = None
        if hasattr(self, 'tokenizer') and self.tokenizer is not None: del self.tokenizer; self.tokenizer = None
        self.loaded = False
        gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    @torch.inference_mode()
    def _infer_model_specific(self, images, questions):
        # No compilation logic is needed. We run directly in eager mode.
            
        final_results = [{"caption": "", "answer": ""} for _ in range(len(images))]
        
        for i, img in enumerate(images):
            try:
                # 1. Captioning
                final_results[i]['caption'] = self.model.answer_question(
                    image=img,
                    question="Describe this image in detail.",
                    tokenizer=self.tokenizer
                ).strip()

                # 2. VQA
                if self.config['models_vqa_enabled'].get('moondream'):
                    q = questions[i]
                    if q and q.strip():
                        final_results[i]['answer'] = self.model.answer_question(
                            image=img,
                            question=q,
                            tokenizer=self.tokenizer
                        ).strip()
            except Exception as e:
                self._log(f"Error during Moondream inference on one image: {e}")
                final_results[i]['caption'] = f"Error: {e}"
                final_results[i]['answer'] = f"Error: {e}"

        return final_results