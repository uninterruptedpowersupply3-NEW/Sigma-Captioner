import os
import json
import time
import traceback
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
import gc
import sys
import requests
import asyncio
import platform
# Set Windows Selector Event Loop Policy for high-concurrency aiohttp performance
if platform.system() == 'Windows':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except:
        pass
import aiohttp
import base64
import io
import re
import concurrent.futures
import queue
import threading
import glob
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Set Windows Selector Event Loop Policy for high-concurrency aiohttp performance
if platform.system() == 'Windows':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except:
        pass

try:
    from PyQt6.QtCore import QObject, pyqtSignal, QThread
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    class QObject: pass
    class QThread: pass
    def pyqtSignal(*args, **kwargs):
        class MockSignal:
            def emit(self, *args, **kwargs): pass
        return MockSignal()
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
        self.url = self.config.get("sglang_url") or self.config.get("sglang_endpoint") or "http://127.0.0.1:30000/generate"
        self.concurrency = self.config.get("sglang_concurrency", 40)
        self.max_tokens = self.config.get("sglang_max_tokens", 1024)
        self.max_res = self.config.get("sglang_max_res", 256)
        self.system_context = self.config.get("sglang_system_context", "")
        self._resize_logged = False
        self._session = None
        self._connector = None

    def load(self):
        self._log(f"SGLang Client Ready: {self.url}")
        
        # Auto-Start via WSL logic (Simplified)
        if self.config.get("sglang_auto_wsl", False):
            try:
                test_url = self.url.replace('/generate', '/v1/models').replace('/v1/chat/completions', '/v1/models')
                resp = requests.get(test_url, timeout=2)
                if resp.status_code == 200:
                    self._log("[SGLang] Server already responding.")
                    return
            except: pass

            wsl_cmd = self.config.get("sglang_wsl_cmd", "").strip()
            if wsl_cmd:
                self._log(f"[SGLang] Starting WSL server: {wsl_cmd}")
                import subprocess
                cmd = wsl_cmd if wsl_cmd.lower().startswith("wsl") else ["wsl", "bash", "-c", wsl_cmd]
                try:
                    subprocess.Popen(cmd, shell=isinstance(cmd, str), creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
                    time.sleep(5) 
                except Exception as e:
                    self._log(f"[SGLang] Failed to launch WSL server: {e}")

    def unload(self):
        self._session = None
        self._connector = None
        
        if self.config.get("sglang_shutdown_wsl_on_unload", True):
            try:
                import subprocess
                subprocess.run(["wsl", "--shutdown"], check=False)
            except: pass
            time.sleep(10)
        
    def _infer_model_specific(self, images, **kwargs):
        async def run_burst():
            # High-throughput session + Expert Semaphore logic from SGLangPro
            import aiohttp
            # Expert Optimization: Persistent Connector with a slight headroom over concurrency
            connector = aiohttp.TCPConnector(limit=self.concurrency + 20, force_close=False)
            timeout = aiohttp.ClientTimeout(total=600)
            sem = asyncio.Semaphore(self.concurrency)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                tasks = [self._async_infer(session, sem, img) for img in images]
                return await asyncio.gather(*tasks)
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                return executor.submit(lambda: asyncio.run(run_burst())).result()
        else:
            return loop.run_until_complete(run_burst())

    async def _async_infer(self, session, sem, img):
        async with sem:
            try:
                if not img:
                    return {"error": "Invalid image"}

                # Optimized Image Prep from SGLangPro: Expert threading avoids GIL/IO bottlenecks
                def prepare_image(img_obj):
                    if img_obj.mode != 'RGB':
                        img_obj = img_obj.convert('RGB')
                    orig_w, orig_h = img_obj.size
                    new_w, new_h = orig_w, orig_h
                    if orig_w > self.max_res or orig_h > self.max_res:
                        img_obj.thumbnail((self.max_res, self.max_res), Image.Resampling.LANCZOS)
                        new_w, new_h = img_obj.size
                    buf = io.BytesIO()
                    img_obj.save(buf, format="JPEG", quality=85)
                    raw_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    return f"data:image/jpeg;base64,{raw_b64}", orig_w, orig_h, new_w, new_h
                
                # Expert logic: Offload PIL decoding and resizing to a thread pool for maximum throughput
                raw_b64_str, orig_w, orig_h, new_w, new_h = await asyncio.to_thread(prepare_image, img)
                
                if not self._resize_logged:
                    if orig_w > self.max_res or orig_h > self.max_res:
                        self._log(f"[SGLang] Scaled image (MAX dim {self.max_res}): {orig_w}x{orig_h} -> {new_w}x{new_h}")
                        self._resize_logged = True

                # Endpoint Routing
                target_url = self.url
                if "/generate" in target_url:
                    target_url = target_url.replace("/generate", "/v1/chat/completions")
                elif not target_url.endswith("/v1/chat/completions"):
                    target_url = target_url.rstrip("/") + "/v1/chat/completions"

                # JSON Schema Grammar Logic from Pro-GUI
                JSON_SCHEMA = {
                    "type": "object",
                    "properties": {
                        "thought_process": {"type": "string", "description": "Think step-by-step here. Keep it brief."},
                        "question": {"type": "string", "description": "Concise, highly technical question about the image."},
                        "answer": {"type": "string", "description": "Long, detailed, and technically exhaustive answer."}
                    },
                    "required": ["thought_process", "question", "answer"],
                    "additionalProperties": False
                }

                payload = {
                    "model": "default",
                    "messages": [
                        {"role": "system", "content": self.system_context},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": raw_b64_str}},
                            {"type": "text", "text": "Analyze this image based on your character instructions."}
                        ]}
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "qwen_extraction_schema",
                            "strict": True,
                            "schema": JSON_SCHEMA
                        }
                    },
                    "max_tokens": self.max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "presence_penalty": 1.5,
                    "stop": ["</answer>"]
                }
                
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        async with session.post(target_url, json=payload) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                generated_text = data["choices"][0]["message"].get("content", "")
                                return self._parse_output(generated_text)
                            if resp.status in (503, 429) and attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return {"error": f"HTTP {resp.status}"}
                    except Exception as e:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return {"error": str(e)}
                return {"error": "Maximum retries exceeded"}
            except Exception as e:
                return {"error": f"SGLang Internal Error: {str(e)}"}

    def _parse_output(self, text, user_prompt="Generative Q&A"):
        question = "What is in this image?"
        thought = ""
        result_answer = text
        try:
            clean_text = text
            if clean_text.startswith("```json"):
                clean_text = clean_text.replace("```json\n", "", 1).replace("```", "")
            data = json.loads(clean_text)
            thought = data.get("thought_process", "").strip()
            question = data.get("question", "").strip()
            result_answer = data.get("answer", "").strip()
        except:
            q_match = re.search(r"<question>([\s\S]*?)(?:</question>|$)", text)
            if q_match: question = q_match.group(1).strip()
            r_match = re.search(r"<reasoning>([\s\S]*?)(?:</reasoning>|$)", text)
            if r_match: thought = r_match.group(1).strip()
            a_match = re.search(r"<answer>([\s\S]*?)(?:</answer>|$)", text)
            if a_match: result_answer = a_match.group(1).strip()

        final_answer_value = f"<reasoning> {thought} </reasoning> <answer> {result_answer} </answer>"
        return {"qa_pairs": [{"question": question, "answer": final_answer_value}]}

class ProcessingWorker(QThread):
    finished = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)
    log = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._is_running = True
        self.write_buffer = {}  # Buffering writes to save NVME endurance
        self.flush_interval = 5000 # Flush every 5000 images for TPU deployment
        
        # Firehose Discovery Variables
        self.path_queue = queue.Queue(maxsize=10000) 
        self.discovery_complete = False
        self.total_images_discovered = 0
        
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

    def run(self):
        # Firehose Logic: The sequential batcher will now handle starting discovery for each model
        self.run_sequential_batched()

    def _discovery_firehose(self):
        formats = ('.jpg', '.jpeg', '.png', '.webp', '.avif')
        image_dir = self.config.get('image_dir')
        if not image_dir or not os.path.isdir(image_dir):
            self.discovery_complete = True
            return

        import zipfile
        try:
            for root, _, files in os.walk(image_dir):
                if not self._is_running: break
                for f in files:
                    ext = f.lower()
                    if ext.endswith(formats):
                        path = os.path.join(root, f)
                        self.path_queue.put(path)
                        self.total_images_discovered += 1
                    elif ext.endswith('.zip'):
                        zip_path = os.path.join(root, f)
                        try:
                            with zipfile.ZipFile(zip_path, 'r') as z:
                                for z_info in z.infolist():
                                    if not self._is_running: break
                                    if not z_info.is_dir() and z_info.filename.lower().endswith(formats):
                                        path = f"{zip_path}::ZIP::{z_info.filename}"
                                        self.path_queue.put(path)
                                        self.total_images_discovered += 1
                        except Exception as e:
                            self.log.emit(f"Failed to read zip {zip_path}: {e}")
        except Exception as e:
            self.log.emit(f"Discovery Firehose Error: {e}")
        finally:
            self.discovery_complete = True
            self.log.emit(f"Discovery Firehose Finished. Total images found: {self.total_images_discovered}")

    def _read_completed_jsonl(self, json_key):
        # RESUME LOGIC: Check for existence of sidecar .json files in the image directory
        # This is more robust than parsing a single massive JSONL
        completed = set()
        return completed # We will handle path-level check inside the loop effectively

    def run_sequential_batched(self):
        import time
        start_time = time.time()
        self.log.emit("--- Starting streaming sequential batch processing ---")
        
        enabled = self.config.get("models_enabled", {})
        # SGLang Legacy / Qwen Legacy logic
        if self.config.get("sglang_legacy_support", False) and enabled.get("smolvlm"):
            self.log.emit("[Config] SGLang legacy ON; remapping to smolvlm keys.")
            enabled["smolvlm"] = False
        
        processed_count_global = 0

        for model_key, model_class in self.model_map.items():
            if not self._is_running: break
            json_key = self.json_key_map[model_key]
            if not enabled.get(json_key): continue
            
            # START FIREHOSE FOR THIS MODEL
            self.discovery_complete = False
            self.total_images_discovered = 0
            # Clear any leftover paths from previous model runs
            while not self.path_queue.empty():
                try: self.path_queue.get_nowait()
                except: break

            discovery_thread = threading.Thread(target=self._discovery_firehose, daemon=True)
            discovery_thread.start()

            model_path = get_model_path(model_key, self.config)
            
            self.log.emit(f"[{model_key}] Loading from {model_path}...")
            try:
                model = model_class(model_key, model_path, self.config)
                model.set_log_callback(self.log.emit)
                model.load()
            except Exception as e:
                self.log.emit(f"[{model_key}] ERROR: {e}")
                continue

            concurrency = self.config.get("sglang_concurrency", 40)
            batch_size = 4 * concurrency if model_key == "SGLang" else self.config.get("model_specific_batch_sizes", {}).get(json_key, 1)
            
            self.log.emit(f"[{model_key}] Processing in firehose windows of size {batch_size}...")

            iteration = 0
            while self._is_running:
                chunk = []
                while len(chunk) < batch_size and (not self.discovery_complete or not self.path_queue.empty()):
                    try:
                        chunk.append(self.path_queue.get(timeout=1))
                    except queue.Empty:
                        if self.discovery_complete: break
                        continue
                
                if not chunk:
                    if self.discovery_complete: break
                    continue

                # Resuming logic
                if self.config.get("resume_processing", True):
                    def _is_done(p):
                        json_sidecar = os.path.splitext(p)[0] + ".json"
                        if not os.path.exists(json_sidecar):
                            return False
                        try:
                            with open(json_sidecar, 'r', encoding='utf-8') as jf:
                                sidecar_data = json.load(jf)
                            check_key = json_key
                            if json_key == "sglang" and self.config.get("sglang_legacy_support", False):
                                check_key = "smolvlm"
                            elif json_key == "qwen" and self.config.get("qwen_legacy_support", False):
                                check_key = "wd_tagger"
                            model_result = sidecar_data.get(check_key)
                            return model_result is not None and not (isinstance(model_result, dict) and "error" in model_result)
                        except Exception:
                            return False
                    chunk = [p for p in chunk if not _is_done(p)]

                if not chunk: continue
                
                iteration += len(chunk)
                processed_count_global += len(chunk)
                
                # Update progress with what we know
                current_total = max(iteration, self.total_images_discovered)
                self.progress.emit(iteration, current_total, f"[{model_key}] Processing window...")

                imgs = []
                valid_paths = []
                for p in chunk:
                    try:
                        if "::ZIP::" in p:
                            zip_path, internal_path = p.split("::ZIP::")
                            import zipfile
                            with zipfile.ZipFile(zip_path, 'r') as z:
                                with z.open(internal_path) as zf:
                                    img_data = zf.read()
                            Image.MAX_IMAGE_PIXELS = None
                            img = Image.open(io.BytesIO(img_data))
                            w, h = img.size
                            max_pixels = 4194304  # 4-Megapixel buffer (2048x2048 eq)
                            if w * h > max_pixels:
                                scale = (max_pixels / (w * h)) ** 0.5
                                img.thumbnail((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                            imgs.append(img.copy())
                            valid_paths.append(p)
                        else:
                            Image.MAX_IMAGE_PIXELS = None
                            img = Image.open(p)
                            # FAST SKIP: Don't do heavy sequential resizing here if SGLang (async/concurrent) handles it.
                            if json_key != "sglang":
                                w, h = img.size
                                max_pixels = 4194304  # 4-Megapixel limit
                                if w * h > max_pixels:
                                    scale = (max_pixels / (w * h)) ** 0.5
                                    img.thumbnail((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                            imgs.append(img.copy())
                            valid_paths.append(p)
                    except Exception as e:
                        self.log.emit(f"Failed to read image {p}: {e}")
                
                if not imgs: continue
                
                try:
                    # Provide questions/prompts if model needs them
                    batch_questions = [self.config.get("common_question", "Describe this image.")] * len(imgs)
                    
                    results = model.infer(imgs, questions=batch_questions, prompts=batch_questions)
                    for j, res in enumerate(results):
                        self._save_result_buffered(valid_paths[j], json_key, res)
                    
                    if len(self.write_buffer) >= self.flush_interval:
                        self._flush_buffer()
                except Exception as e:
                    self.log.emit(f"[{model_key}] Window Error: {e}")

            self._flush_buffer() # Final flush
            model.unload()
            del model
            gc.collect()
            if HAS_TORCH and torch.cuda.is_available(): torch.cuda.empty_cache()
            
            self.log.emit(f"Waiting 10s for {model_key} cleanup...")
            time.sleep(10)

        elapsed = (time.time() - start_time) / 3600.0 # hours
        self.finished.emit(f"Processing Complete. Total Time: {elapsed:.2f} hours")

    def _save_result_buffered(self, image_path, json_key, result):
        # Skip saving error results — they should be retried, not persisted
        if isinstance(result, dict) and "error" in result:
            self.log.emit(f"[{json_key}] Error for {os.path.basename(image_path)}: {result['error'][:100]}")
            return
        if image_path not in self.write_buffer:
            self.write_buffer[image_path] = []
        self.write_buffer[image_path].append((json_key, result))

    def _flush_buffer(self):
        if not self.write_buffer:
            return
        
        count = len(self.write_buffer)
        self.log.emit(f"Flushing {count} sidecar JSON files...")
        
        for image_path, updates in self.write_buffer.items():
            if "::ZIP::" in image_path:
                continue  # Skip ZIP entries — no sidecar for those
            
            json_path = os.path.splitext(image_path)[0] + ".json"
            base_filename = os.path.basename(image_path)
            
            # Check for existing sidecar to merge (multi-model support)
            current_data = {}
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        current_data = json.load(f)
                except Exception:
                    pass
            
            current_data["image_path"] = image_path
            current_data["image_filename"] = base_filename
            current_data["question_used_for_image"] = self.config.get("common_question", "N/A")

            for json_key, result in updates:
                save_key = json_key
                if json_key == "sglang" and self.config.get("sglang_legacy_support", False):
                    save_key = "smolvlm"
                elif json_key == "qwen" and self.config.get("qwen_legacy_support", False):
                    save_key = "wd_tagger"
                
                current_data[save_key] = result
            
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(current_data, f, indent=4, ensure_ascii=False)
            except Exception as e:
                self.log.emit(f"Sidecar Write Error for {base_filename}: {e}")
                
        self.write_buffer.clear()
        self.log.emit(f"Flush complete ({count} files). Cleaning memory...")
        
        # Periodic memory cleanup
        gc.collect()
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def _save_result(self, image_path, json_key, result):
        # Sigma-Captioner Standard: Save individual JSON sidecars
        try:
            if "::ZIP::" not in image_path:
                json_path = os.path.splitext(image_path)[0] + ".json"
                
                # Check for existing data to merge or create new
                current_data = {}
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        current_data = json.load(f)
                
                current_data["image_path"] = image_path
                current_data["image_filename"] = os.path.basename(image_path)
                
                # Extract and format the result
                save_key = json_key
                if json_key == "sglang" and self.config.get("sglang_legacy_support", False):
                    save_key = "smolvlm"
                elif json_key == "qwen" and self.config.get("qwen_legacy_support", False):
                    save_key = "wd_tagger"
                
                current_data[save_key] = result
                
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(current_data, f, indent=4, ensure_ascii=False)
                    
        except Exception as e:
            self.log.emit(f"Sidecar Error for {image_path}: {e}")

        # No buffered JSONL flushes needed here anymore as per request

