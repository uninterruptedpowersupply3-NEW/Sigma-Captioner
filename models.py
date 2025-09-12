# models.py
# Corrected and enhanced wrapper classes for each AI model for batch processing.

import torch
import gc
import os
import json
import random
from PIL import Image
from transformers import (
    AutoProcessor, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration,
    BlipForQuestionAnswering, GitForCausalLM
)
from clip_interrogator import Config, Interrogator
import pandas as pd
import numpy as np
from typing import Dict, Any


try:
    from llama_cpp import Llama
    import tempfile
except ImportError:
    Llama, tempfile = None, None
    print("llama-cpp-python library not available. JoyCaption model will be disabled.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

class BaseModelWrapper:
    def __init__(self, model_key, model_path, config):
        self.model_key, self.model_path, self.config = model_key, model_path, config
        self.model, self.processor, self.loaded, self.log_callback = None, None, False, None

    def set_log_callback(self, callback): self.log_callback = callback
    def _log(self, msg): (self.log_callback or print)(f"[{self.model_key}] {msg}")

    def load(self):
        if self.loaded: return
        self._log(f"Loading from {self.model_path}...")
        try: self._load_model_specific(); self.loaded = True; self._log("Model loaded.")
        except Exception as e: self.unload(); self._log(f"ERROR loading: {e}"); raise

    def unload(self):
        if not self.loaded and not self.model and not self.processor:
            return
        self._log("Unloading...")
        del self.model
        del self.processor
        self.model, self.processor, self.loaded = None, None, False
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        self._log("Unloaded.")

    @torch.no_grad()
    def infer(self, images, **kwargs):
        if not self.loaded: return [f"Error: {self.model_key} not loaded."] * len(images)
        try: return self._infer_model_specific(images, **kwargs)
        except Exception as e: self._log(f"ERROR inference: {e}"); raise

    def _load_model_specific(self): raise NotImplementedError
    def _infer_model_specific(self, images, **kwargs): raise NotImplementedError


# --- FINAL BlipModel CLASS ---
class BlipModel(BaseModelWrapper):
    def __init__(self, model_key, model_path, config):
        super().__init__(model_key, model_path, config)
        self.model_path_cap = model_path
        models_base_dir = os.path.dirname(model_path)
        self.model_path_vqa = os.path.join(models_base_dir, "BLIP_VQA")
        
        self.model_cap, self.processor_cap, self.model_vqa, self.processor_vqa = [None] * 4
        
        self.cap_attn_impl = "N/A"
        self.vqa_attn_impl = "N/A"
        self.is_compiled = False

    def _load_model_specific(self):
        # Logic for the BLIP Captioning model
        self._log("Loading BLIP captioning model...")
        self.processor_cap = BlipProcessor.from_pretrained(self.model_path_cap)
        try:
            self.model_cap = BlipForConditionalGeneration.from_pretrained(
                self.model_path_cap, torch_dtype=DTYPE, attn_implementation="flash_attention_2"
            ).to(DEVICE)
            self.cap_attn_impl = "Flash Attention 2"
        except (ValueError, ImportError):
            self._log("Flash Attention 2 not available for captioner.")
            try:
                self.model_cap = BlipForConditionalGeneration.from_pretrained(
                    self.model_path_cap, torch_dtype=DTYPE, attn_implementation="sdpa"
                ).to(DEVICE)
                self.cap_attn_impl = "SDPA"
            except (ValueError, ImportError):
                self._log("SDPA not available for captioner.")
                try:
                    self.model_cap = BlipForConditionalGeneration.from_pretrained(
                        self.model_path_cap, torch_dtype=DTYPE, attn_implementation="xformers"
                    ).to(DEVICE)
                    self.cap_attn_impl = "Xformers"
                except (ValueError, ImportError):
                    self._log("Xformers not available for captioner, falling back to Eager.")
                    self.model_cap = BlipForConditionalGeneration.from_pretrained(
                        self.model_path_cap, torch_dtype=DTYPE, attn_implementation="eager"
                    ).to(DEVICE)
                    self.cap_attn_impl = "Eager (Default)"
        
        # Logic for the BLIP VQA model
        self._log("Loading BLIP VQA model...")
        self.processor_vqa = BlipProcessor.from_pretrained(self.model_path_vqa)
        try:
            self.model_vqa = BlipForQuestionAnswering.from_pretrained(
                self.model_path_vqa, torch_dtype=DTYPE, attn_implementation="flash_attention_2"
            ).to(DEVICE)
            self.vqa_attn_impl = "Flash Attention 2"
        except (ValueError, ImportError):
            self._log("Flash Attention 2 not available for VQA.")
            try:
                self.model_vqa = BlipForQuestionAnswering.from_pretrained(
                    self.model_path_vqa, torch_dtype=DTYPE, attn_implementation="sdpa"
                ).to(DEVICE)
                self.vqa_attn_impl = "SDPA"
            except (ValueError, ImportError):
                self._log("SDPA not available for VQA.")
                try:
                    self.model_vqa = BlipForQuestionAnswering.from_pretrained(
                        self.model_path_vqa, torch_dtype=DTYPE, attn_implementation="xformers"
                    ).to(DEVICE)
                    self.vqa_attn_impl = "Xformers"
                except (ValueError, ImportError):
                    self._log("Xformers not available for VQA, falling back to Eager.")
                    self.model_vqa = BlipForQuestionAnswering.from_pretrained(
                        self.model_path_vqa, torch_dtype=DTYPE, attn_implementation="eager"
                    ).to(DEVICE)
                    self.vqa_attn_impl = "Eager (Default)"

    def unload(self):
        super().unload()
        del self.model_cap, self.processor_cap, self.model_vqa, self.processor_vqa
        self.model_cap, self.processor_cap, self.model_vqa, self.processor_vqa = None, None, None, None
        self.is_compiled = False

    @torch.inference_mode()
    def _infer_model_specific(self, images, questions):
        if not self.is_compiled:
            use_graphs = self.config.get('use_cuda_graphs', False)
            print("\n--- BLIP First Batch Analysis (One-Time Cost) ---")
            print(f"  > Captioner Attention: {self.cap_attn_impl}")
            print(f"  > VQA Attention:       {self.vqa_attn_impl}")
            
            if use_graphs:
                print("  > Config: CUDA Graphs       = ENABLED")
                self._log("Compiling BLIP models with fullgraph=True...")
                try:
                    self.model_cap = torch.compile(self.model_cap, mode="reduce-overhead", fullgraph=True)
                    self.model_vqa = torch.compile(self.model_vqa, mode="reduce-overhead", fullgraph=True)
                    print("  > Status: SUCCESS (CUDA Graph Mode)")
                except Exception as e:
                    print(f"  > Status: FAILED. Error: {e}. Falling back to compatibility mode.")
                    self.model_cap = torch.compile(self.model_cap, mode="reduce-overhead")
                    self.model_vqa = torch.compile(self.model_vqa, mode="reduce-overhead")
            else:
                print("  > Config: CUDA Graphs       = DISABLED")
                self._log("Compiling BLIP models for compatibility...")
                self.model_cap = torch.compile(self.model_cap, mode="reduce-overhead")
                self.model_vqa = torch.compile(self.model_vqa, mode="reduce-overhead")
                print("  > Status: SUCCESS (Compatibility Mode)")
            
            print("---------------------------------------------------\n")
            self.is_compiled = True
        
        final_results = [{"caption": "", "answer": ""} for _ in range(len(images))]
        inputs_cap = self.processor_cap(images=images, return_tensors="pt", padding=True).to(DEVICE)
        out_cap = self.model_cap.generate(**inputs_cap, max_new_tokens=self.config['model_specific_max_words'].get('blip', 75))
        captions = self.processor_cap.batch_decode(out_cap, skip_special_tokens=True)
        for i, cap in enumerate(captions):
            final_results[i]["caption"] = cap.strip()

        if self.config['models_vqa_enabled'].get('blip') and questions and any(q and q.strip() for q in questions):
            vqa_indices = [i for i, q in enumerate(questions) if q and q.strip()]
            if vqa_indices:
                vqa_images = [images[i] for i in vqa_indices]
                vqa_questions = [questions[i] for i in vqa_indices]
                inputs_vqa = self.processor_vqa(images=vqa_images, text=vqa_questions, return_tensors="pt", padding=True).to(DEVICE)
                out_vqa = self.model_vqa.generate(**inputs_vqa, max_new_tokens=self.config['model_specific_max_words'].get('blip', 75))
                answers = self.processor_vqa.batch_decode(out_vqa, skip_special_tokens=True)
                for i, ans in enumerate(answers):
                    final_results[vqa_indices[i]]["answer"] = ans.strip()
        return final_results


#b
#l
#i
#p


# --- FINAL FlorenceModel CLASS ---
class FlorenceModel(BaseModelWrapper):
    def __init__(self, model_key: str, model_path: str, config: Dict[str, Any]):
        super().__init__(model_key, model_path, config)
        self.attn_impl = "N/A"
        self.is_compiled = False # Simple flag to ensure compilation happens only once

    def _load_model_specific(self):
        try:
            self._log("Attempting to load Florence-2 with SAGE ATTN 1...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, trust_remote_code=True, torch_dtype=DTYPE,
                attn_implementation="sage_attn_1"
            ).eval().to(DEVICE)
            self.attn_impl = "Sage Attn 1"
        except (ValueError, ImportError):
            self._log("Sage Attn 1 failed. Trying FLASH ATTENTION 2...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, trust_remote_code=True, torch_dtype=DTYPE,
                    attn_implementation="flash_attention_2"
                ).eval().to(DEVICE)
                self.attn_impl = "Flash Attention 2"
            except (ValueError, ImportError):
                self._log("Flash Attention 2 failed. Using default EAGER attention.")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, trust_remote_code=True, torch_dtype=DTYPE,
                    attn_implementation="eager"
                ).eval().to(DEVICE)
                self.attn_impl = "Eager (Default)"

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

        print("\n--- Florence-2 Model Optimizations Loaded ---")
        print(f"  > Precision:                {DTYPE}")
        print(f"  > Attention Implementation: {self.attn_impl}")
        print(f"  > JIT Compilation:          YES (on first batch)")
        print("-------------------------------------------\n")

    def unload(self):
        # Ensure parent unload is called to clear generic model/processor
        super().unload()
        self.is_compiled = False

# models.py

# --- In the FlorenceModel class, REPLACE the _cleanup_parsed_output method ---

# models.py -> In the FlorenceModel class

    # --- REPLACE THIS ENTIRE METHOD ---
    def _cleanup_parsed_output(self, data):
        if isinstance(data, str):
            cleaned_str = data.strip()

            # --- NEW FILTERING LOGIC ---
            if self.config.get('florence_filter_ocr', False):
                import re
                # THE FIX: Replace broad '\w' with specific 'a-zA-Z0-9_'
                pattern = (
                    r'[^a-zA-Z0-9_\s'         # Allow ONLY English alphanumeric + underscore + whitespace
                    r'\u4e00-\u9fff'          # CJK Unified Ideographs (Chinese)
                    r'\u3040-\u309f'          # Japanese Hiragana
                    r'\u30a0-\u30ff'          # Japanese Katakana
                    r'\uff00-\uffef'          # Full-width forms
                    r'\u3000-\u303f'          # CJK Symbols and Punctuation
                    r'.,!?"\'()\[\]{}<>\-+=*/|\\%#@&~`' # Common symbols
                    r']'                      # Close the character set
                )
                # This regex finds all characters NOT in the allowed set and removes them.
                allowed_chars_pattern = re.compile(pattern)
                cleaned_str = allowed_chars_pattern.sub('', cleaned_str)
            # --- END OF FILTERING LOGIC ---

            if cleaned_str.startswith('<VQA>'):
                end_of_prompt_index = cleaned_str.find('>')
                if end_of_prompt_index != -1:
                     cleaned_str = cleaned_str[end_of_prompt_index + 1:].strip()
            
            import re
            cleaned_str = re.sub(r'<loc_\d+>', '', cleaned_str).strip()

            if cleaned_str in ["-", "٠٠"]: return ""
            return cleaned_str.replace('<pad>', '').replace('</s>', '').replace('<s>', '').strip()

        if isinstance(data, dict):
            if 'labels' in data and isinstance(data.get('labels'), list):
                # Recursively filter labels in dense captions
                if self.config.get('florence_filter_ocr', False):
                    data['labels'] = [self._cleanup_parsed_output(label) for label in data['labels']]
                
                # Remove empty labels after filtering
                data['labels'] = [label for label in data['labels'] if label]
                
                if not data['labels']:
                    del data['labels']
            
            return {key: self._cleanup_parsed_output(value) for key, value in data.items()}

        if isinstance(data, list):
            return [self._cleanup_parsed_output(item) for item in data]
        
        return data

    @torch.inference_mode()
    def _infer_model_specific(self, images, **kwargs):
        # ONE-TIME COMPILATION LOGIC (The simple, fast pattern)
        if not self.is_compiled:
            use_graphs = self.config.get('use_cuda_graphs', False)
            print("\n--- Florence-2 First Batch Analysis (One-Time Cost) ---")
            if use_graphs:
                print("  > Config: CUDA Graphs       = ENABLED")
                self._log("Compiling Florence-2 with fullgraph=True...")
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)
                    print("  > Status: SUCCESS (CUDA Graph Mode)")
                except Exception as e:
                    print(f"  > Status: FAILED. Error: {e}. Falling back.")
                    self.model = torch.compile(self.model, mode="reduce-overhead")
            else:
                print("  > Config: CUDA Graphs       = DISABLED")
                self._log("Compiling Florence-2 for compatibility...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("  > Status: SUCCESS (Compatibility Mode)")
            print("-------------------------------------------------------\n")
            self.is_compiled = True
            self._log("Florence-2 JIT compilation complete.")

        # --- INFERENCE LOGIC ---
        prompts = kwargs.get('prompts')
        if not prompts: return [{"error": "No prompts provided."}] * len(images)

        rgb_images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
        
        inputs = self.processor(text=prompts, images=rgb_images, return_tensors="pt", padding=True).to(device=DEVICE, dtype=DTYPE)

        with torch.autocast(device_type="cuda", dtype=DTYPE):
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                max_new_tokens=self.config['model_specific_max_words'].get('florence', 1024),
                do_sample=False, num_beams=1,
            )
        
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
        
        batch_outputs = []
        for i, text in enumerate(generated_texts):
            try:
                # Get the original prompt (handles list or single prompt)
                raw_prompt = prompts[i] if isinstance(prompts, (list, tuple)) else prompts

                # Extract only the leading task token like "<VQA>" (if present)
                import re
                # THE FIX: Add '\s*' to handle optional leading whitespace in prompts.
                m = re.match(r'^\s*(<[^>]+>)', raw_prompt or "")
                task_token = m.group(1) if m else raw_prompt

                # Pass only the task token (not the full question) to the post-processor
                parsed = self.processor.post_process_generation(
                    text,
                    task=task_token,
                    image_size=rgb_images[i].size
                )

                # Use your existing cleanup which handles str/dict/list recursively
                batch_outputs.append(self._cleanup_parsed_output(parsed))
            except Exception as e:
                self._log(f"ERROR post-processing for prompt '{prompts[i] if isinstance(prompts, (list, tuple)) else prompts}': {e}")
                batch_outputs.append({"error": str(e), "raw_output": self._cleanup_parsed_output(text)})
        return batch_outputs



# --- CLIP INTERROGATOR MODEL ---
class ClipInterrogatorModel(BaseModelWrapper):
    def _load_model_specific(self):
        ci_config = Config(clip_model_name=self.config.get('clip_model_variant', 'ViT-L-14/openai')); ci_config.cache_path = self.model_path; ci_config.device = DEVICE
        self.ci = Interrogator(ci_config)
    def unload(self): del self.ci; self.ci = None; super().unload()
    def _infer_model_specific(self, images, **kwargs): return [{"caption": self.ci.interrogate(img)} for img in images]

# --- JOYCAPTION MODEL ---
class JoyCaptionModel(BaseModelWrapper):
    def _load_model_specific(self):
        if Llama is None: raise ImportError("llama-cpp-python is not installed.")
        self.model = Llama(self.model_path, n_gpu_layers=self.config.get('llava_n_gpu_layers', 99), n_ctx=self.config.get('llava_n_ctx', 2048), verbose=False)

    def _infer_model_specific(self, images, questions):
        batch_results = []
        for i, image in enumerate(images):
            results = {}; prompt = "Please describe this image in detail."; results['caption'] = self._generate_caption(image, prompt)
            if self.config['models_vqa_enabled'].get('llava') and questions and questions[i]: results['answer'] = self._generate_caption(image, questions[i])
            batch_results.append(results)
        return batch_results

    def _generate_caption(self, image, prompt):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img: image.save(temp_img.name); temp_path = temp_img.name
        try:
            full_prompt = f"### Instruction: {prompt}\n### Input:\n<img_path>{temp_path}\n\n### Response:"
            output = self.model(full_prompt, max_tokens=self.config['model_specific_max_words'].get('llava', 256), temperature=0.01, top_k=5, top_p=0.9, stop=["###"], echo=False, stream=False)
            return output['choices'][0]['text'].strip()
        except Exception as e: self._log(f"JoyCaption failed: {e}"); return f"Error: {e}"
        finally: os.remove(temp_path)

# --- GIT MODEL ---
# models.py

# ... (keep all other imports and classes like BaseModelWrapper, FlorenceModel, etc.)


# --- REPLACE THE ENTIRE GitModel CLASS WITH THIS FINAL VERSION ---

# In models.py - This is the complete, optimized class to replace the old one.

class GitModel(BaseModelWrapper):
    def __init__(self, model_key, model_path, config):
        super().__init__(model_key, model_path, config)
        self.is_compiled = False
        self.current_compile_mode = 'none'

# In models.py -> class GitModel

    def _load_model_specific(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        try:
            self._log("GIT: Attempting to load with Flash Attention 2...")
            self.model = GitForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=DTYPE,
                attn_implementation="flash_attention_2"
            ).to(DEVICE).eval()
            self.attn_impl = "Flash Attention 2"
            self._log("GIT: Successfully loaded with Flash Attention 2.")
        except (ValueError, ImportError):
            self._log("GIT: Flash Attention 2 not available. Falling back to SDPA.")
            try:
                self.model = GitForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=DTYPE,
                    attn_implementation="sdpa"
                ).to(DEVICE).eval()
                self.attn_impl = "SDPA"
                self._log("GIT: Successfully loaded with SDPA.")
            except (ValueError, ImportError):
                self._log("GIT: SDPA not available. Falling back to Xformers.")
                try:
                    self.model = GitForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=DTYPE,
                        attn_implementation="xformers"
                    ).to(DEVICE).eval()
                    self.attn_impl = "Xformers"
                    self._log("GIT: Successfully loaded with Xformers.")
                except (ValueError, ImportError):
                    self._log("GIT: Xformers not available. Falling back to Eager (default).")
                    self.model = GitForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=DTYPE,
                        attn_implementation="eager"
                    ).to(DEVICE).eval()
                    self.attn_impl = "Eager (Default)"
                    self._log("GIT: Successfully loaded with Eager.")

    def unload(self):
        super().unload()
        self.is_compiled = False
        self.current_compile_mode = 'none'

    def compile_model(self):
        if self.is_compiled: return

        use_graphs_config = self.config.get('use_cuda_graphs', False)
        if use_graphs_config:
            print("\n--- Compiling GIT (Attempting CUDA Graph Mode) ---")
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)
                self.current_compile_mode = 'fullgraph'
                self._log("GIT compilation successful in CUDA Graph mode.")
            except Exception as e:
                self._log(f"GIT CUDA Graph compilation FAILED: {e}. Falling back.")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self.current_compile_mode = 'reduce-overhead'
                self._log("GIT compilation successful in compatibility mode.")
        else:
            print("\n--- Compiling GIT (Compatibility Mode) ---")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            self.current_compile_mode = 'reduce-overhead'
            self._log("GIT compilation successful in compatibility mode.")
        
        self.is_compiled = True
        print("--------------------------------------------------\n")

    @torch.inference_mode()
    def _infer_model_specific(self, images, questions=None):
        if not self.is_compiled:
            self.compile_model()

        # --- FIX: Initialize results with only the caption key ---
        results = [{"caption": ""} for _ in range(len(images))]
        
        rgb_images = [img.convert("RGB") for img in images]
        pixel_values = self.processor(images=rgb_images, return_tensors="pt").pixel_values.to(DEVICE, DTYPE)
        
        # --- Step 1: Captioning ---
        cap_ids = self.model.generate(
            pixel_values=pixel_values,
            max_new_tokens=self.config['model_specific_max_words'].get('git', 75),
            num_beams=1,
            do_sample=False
        )
        captions = self.processor.batch_decode(cap_ids, skip_special_tokens=True)
        for i, cap in enumerate(captions):
            results[i]["caption"] = cap.strip()

        # --- Step 2: VQA (Logic remains the same, but now adds the 'answer' key) ---
        if self.config['models_vqa_enabled'].get('git') and questions and any(q and q.strip() for q in questions):
            vqa_indices = [i for i, q in enumerate(questions) if q and q.strip()]
            if vqa_indices:
                vqa_pixel_values = pixel_values[vqa_indices]
                vqa_questions = [questions[i] for i in vqa_indices]
                input_ids = self.processor(text=vqa_questions, padding=True, return_tensors="pt").input_ids.to(DEVICE)
                
                vqa_ids = self.model.generate(
                    pixel_values=vqa_pixel_values,
                    input_ids=input_ids,
                    max_new_tokens=self.config['model_specific_max_words'].get('git', 75),
                    num_beams=1,
                    do_sample=False
                )
                answers = self.processor.batch_decode(vqa_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
                for i, ans in enumerate(answers):
                    # This line now correctly ADDS the key instead of just updating it
                    results[vqa_indices[i]]["answer"] = ans.strip()

        return results