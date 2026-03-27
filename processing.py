import os
import json
import time
import traceback
import torch
import gc
import sys
import requests
import asyncio
import aiohttp
import base64
import io
import re
import concurrent.futures
from PIL import Image
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from collections import defaultdict
from utils import get_model_path

# Add qwen_embedding to path
qwen_path = os.path.join(os.path.dirname(__file__), 'qwen_embedding')
if qwen_path not in sys.path:
    sys.path.append(qwen_path)

class BaseModelWrapper:
    def __init__(self, model_key, model_path, config):
        self.model_key = model_key
        self.model_path = model_path
        self.config = config
        self.log_callback = print

    def set_log_callback(self, cb):
        self.log_callback = cb

    def _log(self, msg):
        self.log_callback(msg)

    def load(self):
        self._load_model_specific()

    def unload(self):
        self._unload_model_specific()

    def _load_model_specific(self):
        pass

    def _unload_model_specific(self):
        pass

    def infer(self, images, **kwargs):
        return self._infer_model_specific(images, **kwargs)

    def _infer_model_specific(self, images, **kwargs):
        return [{"caption": "Base model output"}] * len(images)

class QwenModel(BaseModelWrapper):
    def _load_model_specific(self):
        from backend import FastQwenEngine
        self.engine = FastQwenEngine(
            model_id=self.model_path,
            precision=self.config.get("qwen_precision", "bf16"),
            tf32=self.config.get("qwen_tf32", True),
            quant=self.config.get("qwen_quant", False),
            output_dim=self.config.get("qwen_output_dim", 2048),
            inductor_cache_dir=self.config.get("qwen_inductor_cache_dir", ""),
            inductor_compile_threads=self.config.get("qwen_inductor_compile_threads", 0),
            log_callback=self.log_callback
        )
        cache_path = self.config.get("qwen_json_cache_path", "")
        if cache_path and os.path.exists(cache_path):
            self._log(f"Loading Qwen vocab cache: {cache_path}")
            self.engine.load_vocab_cache(cache_path)
        else:
            self._log("Notice: No Qwen vocab cache found. Using default tags.")
            self.engine.prepare_vocabulary(["masterpiece", "best quality", "1girl", "1boy", "anime"])
        
    def _infer_model_specific(self, images, **kwargs):
        results = self.engine.predict_batch(
            images,
            threshold=self.config.get("qwen_threshold", 0.3),
            max_tags=self.config.get("qwen_max_tags", 50),
            use_compile=self.config.get("qwen_compile", True),
            dynamic_compile=self.config.get("qwen_dynamic", True),
            use_cuda_graphs=self.config.get("qwen_cuda_graphs", False),
            use_pinned_memory=self.config.get("qwen_pinned_mem", True)
        )
        return [{"caption": ", ".join([t for t, s in res])} for res in results]

    def _unload_model_specific(self):
        if hasattr(self, 'engine'):
            del self.engine
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

class SGLangModel(BaseModelWrapper):
    def __init__(self, model_key, model_path, config):
        super().__init__(model_key, model_path, config)
        # Priority: sglang_url -> sglang_endpoint -> default
        self.url = self.config.get("sglang_url") or self.config.get("sglang_endpoint") or "http://127.0.0.1:30000/generate"
        if "/v1" in self.url and "/generate" not in self.url:
            # If user provided as V1 endpoint, we might need to adjust for internal lookups if code expects /generate
            # But the Client handles both. SGLangModel logic might need a base.
            pass
        self.concurrency = self.config.get("sglang_concurrency", 40)
        self.max_tokens = self.config.get("sglang_max_tokens", 1024)
        self.max_res = self.config.get("sglang_max_res", 256)
        self.system_context = self.config.get("sglang_system_context", "")

    def load(self):
        self._log(f"SGLang Client Ready: {self.url}")
        
        # Auto-Start via WSL
        if self.config.get("sglang_auto_wsl", False):
            # First check if server is already responding
            try:
                # Use /v1/models or similar as a light health check
                test_url = self.url.replace('/generate', '/v1/models').replace('/v1/chat/completions', '/v1/models')
                resp = requests.get(test_url, timeout=2)
                if resp.status_code == 200:
                    self._log("[SGLang] Server already responding. Skipping launch.")
                    return
            except:
                pass

            wsl_cmd = self.config.get("sglang_wsl_cmd", "").strip()
            if wsl_cmd:
                self._log(f"[SGLang] Starting WSL server: {wsl_cmd}")
                import subprocess
                
                # Robust command handling: if it already starts with wsl, run it directly
                if wsl_cmd.lower().startswith("wsl"):
                    if os.name == 'nt':
                        cmd = wsl_cmd # Run as string
                    else:
                        cmd = ["bash", "-c", wsl_cmd]
                else:
                    cmd = ["wsl", "bash", "-c", wsl_cmd]

                try:
                    subprocess.Popen(
                        cmd, 
                        shell=isinstance(cmd, str), 
                        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
                    )
                    
                    # Health check loop
                    self._log("[SGLang] Waiting for server to become responsive...")
                    start_time = time.time()
                    connected = False
                    # Health check URL (remove trailing /generate or /v1 bits)
                    health_url = self.url.replace('/generate', '').split('/v1')[0]
                    
                    while time.time() - start_time < 120: # 2 min timeout
                        try:
                            # Try the root or a known info endpoint
                            # SGLang uses /v1/models or just /
                            requests.get(health_url, timeout=1)
                            connected = True
                            break
                        except:
                            time.sleep(3)
                    
                    if connected:
                        self._log("[SGLang] Server is up and responding!")
                    else:
                        self._log("[SGLang] WARNING: Server failed to respond within 120s. Pipeline may fail.")
                except Exception as e:
                    self._log(f"[SGLang] Failed to launch WSL server: {e}")

    def unload(self):
        should_shutdown = self.config.get("sglang_shutdown_wsl_on_unload", True)
        if should_shutdown:
            self._log("Initiating total shutdown of WSL Host to reclaim System RAM...")
            try:
                import subprocess
                subprocess.run(["wsl", "--shutdown"], check=False)
                self._log("[SGLang] Unload: WSL Shutdown signal sent.")
            except Exception as e:
                self._log(f"WSL shutdown hook failed: {e}")
            
            # Mandatory "settling" delay for hardware reclamation
            self._log("Waiting 10 seconds for WSL buffers to clear and VRAM to stabilize...")
            time.sleep(10)
        
    def _infer_model_specific(self, images, **kwargs):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def run_burst():
            sem = asyncio.Semaphore(self.concurrency)
            async with aiohttp.ClientSession() as session:
                tasks = [self._async_infer(session, sem, img) for img in images]
                return await asyncio.gather(*tasks)
        
        return loop.run_until_complete(run_burst())

    async def _async_infer(self, session, sem, img):
        async with sem:
            try:
                # Resize if needed
                orig_w, orig_h = img.size
                if img.width > self.max_res or img.height > self.max_res:
                    img = img.copy()
                    img.thumbnail((self.max_res, self.max_res), Image.Resampling.LANCZOS)
                
                new_w, new_h = img.size
                if new_w != orig_w or new_h != orig_h:
                    self._log(f"[SGLang] Resizing image for context safety: {orig_w}x{orig_h} -> {new_w}x{new_h}")
                # Quiet mode: no log if it fits perfectly

                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=85)
                b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

                # Context Correction: 
                # If the user has a 1024 context, and max_tokens is 1024, it will CRASH.
                # Qwen2-VL 256x256 is roughly 334 tokens. 
                # We cap max_new_tokens to (context - 450) to ensure safe intake.
                # Assuming context is 1024 (user setting).
                safe_max_tokens = min(self.max_tokens, 1024 - 450) if self.max_tokens >= 512 else self.max_tokens
                if safe_max_tokens != self.max_tokens:
                    self._log(f"[SGLang] Safety Cap: Reduced max_new_tokens from {self.max_tokens} to {safe_max_tokens} to fit 1024 context.")

                if "/generate" in self.url:
                    # Enforce the <reasoning> <answer> format via SGLang regex
                    format_regex = r"<reasoning> [\s\S]*? </reasoning> <answer> [\s\S]*? </answer>"
                    
                    payload = {
                        "text": f"<|im_start|>system\n{self.system_context}<|im_end|>\n<|im_start|>user\n<image>\nPlease analyze.<|im_end|>\n<|im_start|>assistant\n",
                        "image_data": b64_img,
                        "sampling_params": {
                            "temperature": 0.7, 
                            "max_new_tokens": safe_max_tokens,
                            "regex": format_regex
                        }
                    }
                else:
                    payload = {
                        "model": "default",
                        "messages": [
                            {"role": "system", "content": self.system_context},
                            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}, {"type": "text", "text": "Analyze image"}]}
                        ],
                        "max_tokens": safe_max_tokens
                    }
                    if self.config.get("sglang_disable_reasoning", True):
                        payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

                async with session.post(self.url, json=payload, timeout=120) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        text = (data.get("text") or data.get("choices", [{}])[0].get("message", {}).get("content", ""))
                        if not text and "choices" in data:
                             text = data["choices"][0].get("text", "")
                        return self._parse_output(text)
                    
                    err_text = await resp.text()
                    return {"error": f"HTTP {resp.status}: {err_text[:200]}"}
            except Exception as e:
                return {"error": str(e)}

    def _parse_output(self, text):
        # Explicit Tag Extraction for <reasoning> and <answer>
        reasoning = ""
        answer = ""
        
        r_match = re.search(r"<reasoning>([\s\S]*?)</reasoning>", text)
        if r_match:
            reasoning = r_match.group(1).strip()
        
        a_match = re.search(r"<answer>([\s\S]*?)</answer>", text)
        if a_match:
            answer_content = a_match.group(1).strip()
            # Advanced JSON extraction if the model nested it inadvertently
            if "{" in answer_content and "}" in answer_content:
                try:
                    json_match = re.search(r"(\{[\s\S]*\})", answer_content)
                    if json_match:
                        ans_json = json.loads(json_match.group(1))
                        if not reasoning and "thought_process" in ans_json:
                            reasoning = ans_json["thought_process"]
                        if "answer" in ans_json:
                            answer = str(ans_json["answer"])
                        elif "caption" in ans_json:
                            answer = str(ans_json["caption"])
                        else:
                            answer = answer_content
                    else:
                        answer = answer_content
                except:
                    answer = answer_content
            else:
                answer = answer_content

        # Fallback for models without tags (e.g. if regex forcing was disabled)
        if not reasoning and not answer:
            if "<think>" in text:
                match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
                if match:
                    reasoning = match.group(1).strip()
                    text = text.replace(match.group(0), "").strip()
            elif "Thought:" in text:
                t_match = re.search(r"Thought:(.*?)(?:\n\n|$)", text, re.DOTALL)
                if t_match:
                    reasoning = t_match.group(1).strip()
                    text = text.replace(t_match.group(0), "").strip()
            answer = text
        
        formatted_answer = f"<reasoning> {reasoning} </reasoning> <answer> {answer} </answer>"
        
        is_legacy = self.config.get("sglang_legacy_support", False)
        if is_legacy:
            return {"qa_pairs": [{"question": "Describe this image.", "answer": formatted_answer}]}
        
        return {"reasoning": reasoning, "answer": answer, "full_output": formatted_answer}

class ProcessingWorker(QThread):
    finished = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)
    log = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._is_running = True
        self.write_buffer = {}  # Buffering writes to save NVME endurance
        self.flush_interval = 500 # Flush every 500 images
        
        # Delayed imports to avoid circular/early load issues
        from models import BlipModel, FlorenceModel, ClipInterrogatorModel, JoyCaptionModel, GitModel
        from moondream_model import MoondreamModel
        from smolvlm_model import SmolVLMModel
        from wd_tagger_app2 import WDTaggerApp2Model

        self.model_map = {
            "SGLang": SGLangModel, # SGLang first as requested
            "Qwen": QwenModel,
            "CLIP_Interrogator": ClipInterrogatorModel, "BLIP": BlipModel,
            "Florence-2": FlorenceModel, "JoyCaption": JoyCaptionModel, "GIT": GitModel,
            "WD_Tagger": WDTaggerApp2Model, "Moondream": MoondreamModel,
            "SmolVLM": SmolVLMModel
        }
        self.json_key_map = {
            "SGLang": "sglang",
            "Qwen": "qwen",
            "CLIP_Interrogator": "clip_interrogator", "BLIP": "blip",
            "Florence-2": "florence", "JoyCaption": "llava", "GIT": "git",
            "WD_Tagger": "wd_tagger", "Moondream": "moondream",
            "SmolVLM": "smolvlm"
        }

    def stop(self):
        self._is_running = False

    def run(self):
        images = self._discover_images()
        if not images:
            self.finished.emit("No images found to process.")
            return
        self.run_sequential_batched(images)

    def _discover_images(self):
        formats = ('.jpg', '.jpeg', '.png', '.webp', '.avif')
        paths = []
        image_dir = self.config.get('image_dir')
        if not image_dir or not os.path.isdir(image_dir):
            return []
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(formats):
                    paths.append(os.path.join(root, f))
        return sorted(paths)

    def run_sequential_batched(self, images_to_process):
        import time
        start_time = time.time()
        self.log.emit("--- Starting sequential batch processing ---")
        total_images = len(images_to_process)
        
        enabled = self.config.get("models_enabled", {})
        # SGLang Legacy / Qwen Legacy logic
        if self.config.get("sglang_legacy_support", False) and enabled.get("smolvlm"):
            self.log.emit("[Config] SGLang legacy ON; remapping to smolvlm keys.")
            enabled["smolvlm"] = False
        
        for model_key, model_class in self.model_map.items():
            if not self._is_running: break
            json_key = self.json_key_map[model_key]
            if not enabled.get(json_key): continue
            
            model_path = get_model_path(model_key, self.config)
            
            # --- MODEL-LEVEL PRE-FILTERING ---
            # Determine if ANY images in the entire set need this specific model
            # before we even attempt to load it. 
            # This prevents the expensive Load/Unload cycle (and WSL shutdown) if not needed.
            any_needed = False
            resume = self.config.get("resume_processing", True)
            if not resume:
                any_needed = True
            else:
                out_dir = self.config.get("output_dir", "output")
                for p in images_to_process:
                    json_path = os.path.join(out_dir, os.path.basename(p) + ".json")
                    if not os.path.exists(json_path):
                        any_needed = True; break
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        target_key = json_key
                        if json_key == "sglang" and self.config.get("sglang_legacy_support", False):
                            target_key = "smolvlm"
                        if target_key not in data:
                            any_needed = True; break
                    except:
                        any_needed = True; break
            
            if not any_needed:
                self.log.emit(f"[{model_key}] Skipping model: All images already processed.")
                continue

            self.log.emit(f"[{model_key}] Loading from {model_path}...")
            try:
                model = model_class(model_key, model_path, self.config)
                model.set_log_callback(self.log.emit)
                model.load()
            except Exception as e:
                self.log.emit(f"[{model_key}] ERROR: {e}")
                continue

            batch_size = self.config.get("model_specific_batch_sizes", {}).get(json_key, 1)
            for i in range(0, len(images_to_process), batch_size):
                if not self._is_running: break
                chunk = images_to_process[i:i+batch_size]
                
                # Resuming logic
                if self.config.get("resume_processing", True):
                    needed = []
                    for p in chunk:
                        out_dir = self.config.get("output_dir", "output")
                        json_path = os.path.join(out_dir, os.path.basename(p) + ".json")
                        if not os.path.exists(json_path):
                            needed.append(p)
                            continue
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            # Check if the specific model's output is missing or incomplete
                            target_key = json_key
                            if json_key == "sglang" and self.config.get("sglang_legacy_support", False):
                                target_key = "smolvlm"
                            if target_key not in data:
                                needed.append(p)
                        except:
                            needed.append(p)
                    chunk = needed

                if not chunk: continue

                imgs = []
                valid_paths = []
                for p in chunk:
                    try:
                        imgs.append(Image.open(p))
                        valid_paths.append(p)
                    except: pass
                
                if not imgs: continue
                
                self.progress.emit(i, total_images, f"[{model_key}] Processing batch {i//batch_size + 1}...")
                try:
                    # Provide questions/prompts if model needs them
                    batch_questions = [self.config.get("common_question", "Describe this image.")] * len(imgs)
                    
                    results = model.infer(imgs, questions=batch_questions, prompts=batch_questions)
                    for j, res in enumerate(results):
                        self._save_result_buffered(valid_paths[j], json_key, res)
                    
                    if len(self.write_buffer) >= self.flush_interval:
                        self._flush_buffer()
                except Exception as e:
                    self.log.emit(f"[{model_key}] Batch Error: {e}")

            self._flush_buffer() # Final flush
            model.unload()
            del model
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
            self.log.emit(f"Waiting 10s for {model_key} cleanup...")
            time.sleep(10)

        elapsed = (time.time() - start_time) / 3600.0 # hours
        self.finished.emit(f"Processing Complete. Total Time: {elapsed:.2f} hours")

    def _save_result_buffered(self, image_path, json_key, result):
        if image_path not in self.write_buffer:
            self.write_buffer[image_path] = []
        self.write_buffer[image_path].append((json_key, result))

    def _flush_buffer(self):
        if not self.write_buffer:
            return
        
        count = len(self.write_buffer)
        self.log.emit(f"Flushing write buffer for {count} images to disk...")
        
        out_dir = self.config.get("output_dir", "output")
        os.makedirs(out_dir, exist_ok=True)
        
        for image_path, updates in self.write_buffer.items():
            json_path = os.path.join(out_dir, os.path.basename(image_path) + ".json")
            
            # Load existing
            data = {}
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except: pass
            
            # Metadata update
            data["image_path"] = image_path
            data["image_filename"] = os.path.basename(image_path)
            data["question_used_for_image"] = self.config.get("common_question", "N/A")

            # Apply updates
            for json_key, result in updates:
                save_key = json_key
                if json_key == "sglang" and self.config.get("sglang_legacy_support", False):
                    save_key = "smolvlm"
                    if isinstance(result, dict) and "full_output" in result:
                        # Remap to smolvlm schema if not already
                        result = {"qa_pairs": [{"question": "Describe this image.", "answer": result["full_output"]}]}
                elif json_key == "qwen" and self.config.get("qwen_legacy_support", False):
                    save_key = "wd_tagger"
                
                data[save_key] = result

            # Batch write
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
            except Exception as e:
                self.log.emit(f"Disk Write Error: {e}")
                
        self.write_buffer.clear()
        self.log.emit(f"Disk flush complete. Cleaning memory...")
        
        # Periodic memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def _save_result(self, image_path, json_key, result):
        self._save_result_buffered(image_path, json_key, result)
        self._flush_buffer()

