import base64
import io
import json
import os
import sys
import threading
import traceback
from pathlib import Path
import concurrent.futures

from PyQt6.QtCore import QThread, pyqtSignal
from utils import get_model_path


def _ensure_qwen_embedding_on_path() -> str:
    qwen_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_embedding")
    if qwen_dir not in sys.path:
        sys.path.append(qwen_dir)
    return qwen_dir


class _SignalWriter:
    def __init__(self, signal: pyqtSignal):
        self._signal = signal
        self._buffer = ""

    def write(self, s: str):
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                self._signal.emit(line)

    def flush(self):
        if self._buffer.strip():
            self._signal.emit(self._buffer.strip())
        self._buffer = ""


class QwenVocabCacheWorker(QThread):
    log = pyqtSignal(str)

    def __init__(self, danbooru_json_path: str, min_occurrences: int, batch_size: int, use_english_dict: bool, out_pt_path: str, out_json_path: str):
        super().__init__()
        self.danbooru_json_path = danbooru_json_path
        self.min_occurrences = int(min_occurrences)
        self.batch_size = int(batch_size)
        self.use_english_dict = bool(use_english_dict)
        self.out_pt_path = out_pt_path
        self.out_json_path = out_json_path

    def run(self):
        _ensure_qwen_embedding_on_path()

        try:
            from vocab_generator import VocabCacheGenerator

            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = _SignalWriter(self.log)
            sys.stderr = _SignalWriter(self.log)
            try:
                self.log.emit("Starting hybrid vocab extraction...")
                generator = VocabCacheGenerator(precision="int8")
                generator.build_hybrid_danbooru_english_vocab(
                    danbooru_json_path=self.danbooru_json_path,
                    out_pt_path=self.out_pt_path,
                    out_json_path=self.out_json_path,
                    min_occurrences=self.min_occurrences,
                    use_english_dict=self.use_english_dict,
                    batch_size=self.batch_size,
                )
                self.log.emit("Vocab generation complete.")
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
        except Exception as e:
            self.log.emit(f"ERROR (vocab cache): {e}")
            self.log.emit(traceback.format_exc())


class QwenLatentExtractWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)

    IMAGE_CACHE_FILENAME = "qwen_image_cache.pt"

    def __init__(self, image_dir: str, config: dict):
        super().__init__()
        self.image_dir = image_dir
        self.config = config or {}
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True
        self.log.emit("ABORT SIGNAL RECEIVED. Stalling GPU extraction block...")

    def run(self):
        qwen_dir = _ensure_qwen_embedding_on_path()

        try:
            import time
            from concurrent.futures import ThreadPoolExecutor

            import torch

            from backend import FastQwenEngine
            from helper import GPUImageProcessor

            precision = self.config.get("qwen_precision", "bf16")
            tf32 = bool(self.config.get("qwen_tf32", True))
            quant = bool(self.config.get("qwen_quant", False)) or precision in {"int8", "int4"}
            inductor_cache_dir = (self.config.get("qwen_inductor_cache_dir") or "").strip() or None
            inductor_compile_threads = self.config.get("qwen_inductor_compile_threads")

            use_compile = bool(self.config.get("qwen_compile", True))
            dynamic_compile = bool(self.config.get("qwen_dynamic", True))
            use_cuda_graphs = bool(self.config.get("qwen_cuda_graphs", False))
            use_pinned_memory = bool(self.config.get("qwen_pinned_mem", True))

            batch_size = int(self.config.get("qwen_latent_batch_size", self.config.get("model_specific_batch_sizes", {}).get("qwen", 8)))
            prefetch_threads = int(self.config.get("qwen_prefetch", 2))

            self.log.emit(f"Initializing FastQwenEngine ({precision})...")
            model_id = get_model_path("Qwen", self.config)
            engine = FastQwenEngine(
                model_id=model_id,
                precision=precision,
                tf32=tf32,
                quant=quant,
                inductor_cache_dir=inductor_cache_dir,
                inductor_compile_threads=inductor_compile_threads,
            )
            processor = GPUImageProcessor()

            folder = self.image_dir
            if not os.path.isdir(folder):
                self.log.emit(f"ERROR: image_dir does not exist: {folder}")
                return

            files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            ]
            if not files:
                self.log.emit("No images found.")
                return

            self.log.emit(f"Scanning {len(files)} images via multiprocessing pool...")
            t0 = time.time()
            self.progress.emit(0, len(files), "Starting dimensions scan...")
            groups = processor.peek_and_group(
                files, 
                progress_callback=lambda idx, tot, m: self.progress.emit(idx, tot, m),
                log_callback=lambda m: self.log.emit(m)
            )
            self.log.emit(f"Grouped into {len(groups)} uniform resolution blocks in {time.time()-t0:.2f}s.")

            total = 0
            threads = max(1, prefetch_threads)
            all_rel_paths: list[str] = []
            embedding_chunks: list[torch.Tensor] = []

            def write_tensors(valid_paths, pil_images):
                nonlocal total
                if not pil_images or self.is_cancelled:
                    return

                tensor_outputs = engine.extract_image_features(
                    pil_images,
                    use_compile=use_compile,
                    dynamic_compile=dynamic_compile,
                    use_cuda_graphs=use_cuda_graphs,
                    use_pinned_memory=use_pinned_memory,
                )
                embedding_chunks.append(tensor_outputs.detach().cpu())
                all_rel_paths.extend([os.path.relpath(p, folder) for p in valid_paths])
                total += len(valid_paths)
                
                self.progress.emit(total, len(files), f"Extracted latents for {total}/{len(files)} images")
                
                # Directly emit chunk progression without artificial throttling modulo limits
                self.log.emit(f"Accumulated {total}/{len(files)} embeddings (Batch Chunk Mapped)...")
                
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            with ThreadPoolExecutor(max_workers=threads) as executor:
                for target_size, paths in groups.items():
                    self.log.emit(f"Batch mapping shape {target_size} ({len(paths)} images)...")
                    chunks = [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]
                    futures = []

                    for chunk in chunks:
                        if self.is_cancelled:
                            break
                        futures.append(executor.submit(processor.process_group, chunk, target_size))
                        if len(futures) >= threads:
                            valid_paths, pil_images = futures.pop(0).result()
                            write_tensors(valid_paths, pil_images)

                    for fut in futures:
                        if self.is_cancelled:
                            break
                        valid_paths, pil_images = fut.result()
                        write_tensors(valid_paths, pil_images)
                        
                    if self.is_cancelled:
                        break

            if self.is_cancelled:
                self.log.emit("Extraction was cancelled by the user. Partial processing discarded.")
                return

            if not embedding_chunks or not all_rel_paths:
                self.log.emit("No embeddings were generated (all images may have failed to load).")
                return

            embeddings = torch.cat(embedding_chunks, dim=0).contiguous()
            cache_path = os.path.join(folder, self.IMAGE_CACHE_FILENAME)
            torch.save(
                {
                    "paths": all_rel_paths,
                    "embeddings": embeddings,
                },
                cache_path,
            )
            self.log.emit(
                f"Done! Wrote packed image cache -> {cache_path}  (N={embeddings.shape[0]}, D={embeddings.shape[1]}) in {time.time() - t0:.2f}s."
            )
        except Exception as e:
            self.log.emit(f"ERROR (latent extract): {e}")
            self.log.emit(traceback.format_exc())


class QwenFastInferenceWorker(QThread):
    log = pyqtSignal(str)

    IMAGE_CACHE_FILENAME = QwenLatentExtractWorker.IMAGE_CACHE_FILENAME

    def __init__(self, data_dir: str, direct_tags: list[str], json_vocab_path: str, init_type: str, config: dict):
        super().__init__()
        self.data_dir = data_dir
        self.direct_tags = direct_tags or []
        self.json_vocab_path = json_vocab_path
        self.init_type = init_type
        self.config = config or {}

    def run(self):
        _ensure_qwen_embedding_on_path()

        try:
            import time
            from concurrent.futures import ThreadPoolExecutor

            import torch

            from backend import FastQwenEngine
            from helper import GPUImageProcessor

            precision = self.config.get("qwen_precision", "bf16")
            tf32 = bool(self.config.get("qwen_tf32", True))
            quant = bool(self.config.get("qwen_quant", False)) or precision in {"int8", "int4"}
            inductor_cache_dir = (self.config.get("qwen_inductor_cache_dir") or "").strip() or None
            inductor_compile_threads = self.config.get("qwen_inductor_compile_threads")

            threshold = float(self.config.get("qwen_threshold", 0.30))
            max_tags = int(self.config.get("qwen_max_tags", 50))
            batch_size = int(self.config.get("model_specific_batch_sizes", {}).get("qwen", 16))
            prefetch_threads = int(self.config.get("qwen_prefetch", 2))

            use_compile = bool(self.config.get("qwen_compile", True))
            dynamic_compile = bool(self.config.get("qwen_dynamic", True))
            use_cuda_graphs = bool(self.config.get("qwen_cuda_graphs", False))
            use_pinned_memory = bool(self.config.get("qwen_pinned_mem", True))

            if not os.path.isdir(self.data_dir):
                self.log.emit(f"ERROR: data_dir does not exist: {self.data_dir}")
                return

            self.log.emit("Initializing FastQwenEngine. This may take a minute...")
            model_id = get_model_path("Qwen", self.config)
            engine = FastQwenEngine(
                model_id=model_id,
                precision=precision,
                tf32=tf32,
                quant=quant,
                inductor_cache_dir=inductor_cache_dir,
                inductor_compile_threads=inductor_compile_threads,
            )

            if self.init_type == "json":
                if not self.json_vocab_path or not os.path.exists(self.json_vocab_path):
                    self.log.emit(f"ERROR: vocab json not found: {self.json_vocab_path}")
                    return
                self.log.emit(f"Loading massive vocabulary cache from {self.json_vocab_path}...")
                engine.load_vocab_cache(self.json_vocab_path)
                self.log.emit("Engine ready with JSON cache.")
            else:
                vocab = [v.strip() for v in self.direct_tags if v and v.strip()]
                if not vocab:
                    self.log.emit("ERROR: no tags provided for direct-text vocab.")
                    return
                self.log.emit(f"Precomputing cache for {len(vocab)} tags...")
                engine.prepare_vocabulary(vocab)
                self.log.emit("Engine ready.")

            img_files = [
                os.path.join(self.data_dir, f)
                for f in os.listdir(self.data_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            ]
            packed_cache_path = os.path.join(self.data_dir, self.IMAGE_CACHE_FILENAME)
            if os.path.isfile(packed_cache_path):
                self.log.emit(f"Found packed image cache ({os.path.basename(packed_cache_path)}). Running cached inference...")
                cache = torch.load(packed_cache_path, map_location="cpu")
                embeddings = cache.get("embeddings")
                paths = cache.get("paths")
                if embeddings is None or paths is None:
                    self.log.emit("ERROR: packed cache is missing 'embeddings' or 'paths'.")
                    return
                if not hasattr(embeddings, "dim") or embeddings.dim() != 2:
                    self.log.emit(f"ERROR: packed cache embeddings must be 2D [N, D]. Got: {getattr(embeddings, 'shape', None)}")
                    return
                if len(paths) != int(embeddings.shape[0]):
                    self.log.emit(f"ERROR: packed cache paths count ({len(paths)}) != embeddings N ({int(embeddings.shape[0])}).")
                    return

                total_processed = 0
                t0 = time.time()
                chunk_size = max(1024, batch_size * 64)
                for start in range(0, int(embeddings.shape[0]), chunk_size):
                    end = min(start + chunk_size, int(embeddings.shape[0]))
                    batch_latents = embeddings[start:end]
                    results = engine.predict_from_embeddings(batch_latents, threshold=threshold, max_tags=max_tags)
                    for rel_path, res in zip(paths[start:end], results):
                        img_path = rel_path
                        if not os.path.isabs(img_path):
                            img_path = os.path.join(self.data_dir, img_path)
                        clean_tags = ", ".join([t for t, _s in res])
                        txt_path = os.path.splitext(img_path)[0] + ".txt"
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(clean_tags)

                    total_processed += len(results)
                    self.log.emit(f"Processed {total_processed}/{int(embeddings.shape[0])} cached embeddings...")
                    
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                self.log.emit(f"Done! Evaluated {total_processed} cached embeddings in {time.time() - t0:.2f}s.")
                return

            pt_files = [
                os.path.join(self.data_dir, f)
                for f in os.listdir(self.data_dir)
                if f.lower().endswith(".pt") and f != os.path.basename(packed_cache_path)
            ]

            if len(pt_files) > len(img_files) and pt_files:
                self.log.emit(f"Found {len(pt_files)} embedding (.pt) files. Running tensor inference...")
                total_processed = 0
                t0 = time.time()

                chunk_size = max(64, batch_size * 4)
                chunks = [pt_files[i : i + chunk_size] for i in range(0, len(pt_files), chunk_size)]

                for chunk in chunks:
                    tensors = []
                    for pt in chunk:
                        t = torch.load(pt, map_location="cpu")
                        if hasattr(t, "dim"):
                            if t.dim() == 2 and t.shape[0] == 1:
                                t = t[0]
                            if t.dim() != 1:
                                raise ValueError(f"Unexpected embedding shape in {os.path.basename(pt)}: {tuple(t.shape)}")
                        tensors.append(t)

                    batched_latents = torch.stack(tensors, dim=0)
                    results = engine.predict_from_embeddings(batched_latents, threshold=threshold, max_tags=max_tags)

                    total_processed += len(results)
                    for path, res in zip(chunk, results):
                        clean_tags = ", ".join([t for t, _s in res])
                        txt_path = os.path.splitext(path)[0] + ".txt"
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(clean_tags)

                    self.log.emit(f"Processed {total_processed}/{len(pt_files)} embeddings...")
                    
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                t1 = time.time()
                self.log.emit(f"Done! Evaluated {total_processed} embeddings in {t1 - t0:.2f}s.")
                return

            files = img_files
            if not files:
                self.log.emit("No images or embeddings found.")
                return

            self.log.emit(f"Found {len(files)} images. Grouping via peek resizing...")
            processor = GPUImageProcessor()
            groups = processor.peek_and_group(files)
            self.log.emit(f"Grouped into {len(groups)} uniform resolution batches.")

            total_processed = 0
            threads = max(1, prefetch_threads)

            with ThreadPoolExecutor(max_workers=threads) as executor:
                t0 = time.time()
                for target_size, paths in groups.items():
                    self.log.emit(f"Processing shape {target_size} ({len(paths)} images)...")
                    chunks = [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]
                    futures: list[tuple[list[str], object]] = []

                    def flush(chunk_paths, pil_images):
                        nonlocal total_processed
                        if not pil_images:
                            return

                        results = engine.predict_batch(
                            pil_images,
                            threshold=threshold,
                            max_tags=max_tags,
                            use_compile=use_compile,
                            dynamic_compile=dynamic_compile,
                            use_cuda_graphs=use_cuda_graphs,
                            use_pinned_memory=use_pinned_memory,
                        )
                        total_processed += len(results)
                        for path, res in zip(chunk_paths, results):
                            clean_tags = ", ".join([t for t, _s in res])
                            txt_path = os.path.splitext(path)[0] + ".txt"
                            with open(txt_path, "w", encoding="utf-8") as f:
                                f.write(clean_tags)

                    for chunk in chunks:
                        futures.append((chunk, executor.submit(processor.process_group, chunk, target_size)))
                        if len(futures) >= threads:
                            chunk_paths, fut = futures.pop(0)
                            valid_paths, pil_images = fut.result()
                            flush(valid_paths, pil_images)

                    for chunk_paths, fut in futures:
                        valid_paths, pil_images = fut.result()
                        flush(valid_paths, pil_images)

                    self.log.emit(f"Processed {total_processed} images so far...")
                    
                    import gc
                    gc.collect()
                    if hasattr(torch, "cuda") and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                self.log.emit(f"Done! Processed {total_processed} images in {time.time() - t0:.2f}s.")
        except Exception as e:
            self.log.emit(f"ERROR (inference): {e}")
            self.log.emit(traceback.format_exc())


class SGLangStandaloneWorker(QThread):
    log = pyqtSignal(str)

    def __init__(self, config: dict, input_dir: str, output_jsonl_path: str, extension_glob: str):
        super().__init__()
        self.config = config or {}
        self.input_dir = input_dir
        self.output_jsonl_path = output_jsonl_path
        self.extension_glob = extension_glob or "*.jpg"
        self._file_lock = threading.Lock()

    def _get_already_processed(self) -> set[str]:
        processed = set()
        if os.path.exists(self.output_jsonl_path):
            try:
                with open(self.output_jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if "file" in data:
                                processed.add(str(data["file"]))
                        except Exception:
                            continue
            except Exception:
                return set()
        return processed

    def _worker_loop(self, job_queue):
        import requests
        import json
        import time as _time
        from PIL import Image
        import io
        import base64
        
        sglang_url = str(self.config.get("sglang_url", "http://127.0.0.1:30000/generate"))
        chat_url = sglang_url.replace("/generate", "/v1/chat/completions")
        system_context = str(self.config.get("sglang_system_context", "")).strip()
        max_tokens = int(self.config.get("sglang_max_tokens", 1024))
        is_openai = "/v1" in sglang_url
        format_regex = r"<reasoning>[\s\S]*?</reasoning>\s*<answer>[\s\S]*?</answer>"
        
        while True:
            p = job_queue.get()
            if p is None:
                job_queue.task_done()
                break
            
            generated_text = None
            is_error = False
            payload = None
            
            try:
                with Image.open(p) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    # Force longest edge to be exactly 256 for rapid ingestion
                    img.thumbnail((256, 256), Image.Resampling.LANCZOS)
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=85)
                    b64_img_uri = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                if is_openai:
                    payload = {
                        "model": "default",
                        "messages": [
                            {"role": "system", "content": system_context},
                            {"role": "user", "content": [
                                {"type": "image_url", "image_url": {"url": b64_img_uri}},
                                {"type": "text", "text": "<image>\nPlease analyze this image based on the system instructions."}
                            ]}
                        ],
                        # Root-level regex bypasses strict Pydantic literal validation
                        "regex": format_regex,
                        "max_tokens": max_tokens,
                        "temperature": 0.7,
                        "top_p": 0.8,
                        "presence_penalty": 1.5,
                    }
                else:
                    # Native /generate API
                    prompt = f"<|im_start|>system\n{system_context}<|im_end|>\n<|im_start|>user\n<image>\nPlease analyze this image based on the system instructions.<|im_end|>\n<|im_start|>assistant\n"
                    payload = {
                        "text": prompt,
                        "image_data": [b64_img_uri],
                        "sampling_params": {
                            "temperature": 0.7,
                            "max_new_tokens": max_tokens,
                            "regex": format_regex
                        }
                    }
            except Exception as e:
                self.log.emit(f"Failed to prepare {p.name}: {e}")
                is_error = True
                generated_text = f"Preparation error: {e}"
            
            if payload is not None:
                # Retry loop with exponential backoff
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = requests.post(chat_url if is_openai else sglang_url, json=payload, timeout=600)
                        if response.status_code == 200:
                            result = response.json()
                            if is_openai:
                                generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                            else:
                                generated_text = result.get("text", "") if isinstance(result, dict) else ""
                                if not generated_text and isinstance(result, list) and len(result) > 0:
                                    generated_text = result[0].get("text", "")
                            
                            if isinstance(generated_text, str) and generated_text.startswith("```json"):
                                generated_text = generated_text.replace("```json\n", "", 1).replace("```", "")
                            break  # Success
                        elif response.status_code in (503, 429) and attempt < max_retries - 1:
                            _time.sleep(2 ** attempt)
                            continue
                        else:
                            generated_text = f"HTTP {response.status_code}: {response.text[:100]}"
                            is_error = True
                            break
                    except requests.Timeout:
                        if attempt < max_retries - 1:
                            _time.sleep(2 ** attempt)
                            continue
                        generated_text = f"Connection timed out after {max_retries} retries (batch too large or server overloaded?)"
                        is_error = True
                    except (requests.ConnectionError, OSError) as conn_err:
                        if attempt < max_retries - 1:
                            _time.sleep(2 ** attempt)
                            continue
                        generated_text = f"Connection failed after {max_retries} retries: {conn_err}"
                        is_error = True
                    except Exception as e:
                        generated_text = str(e)
                        is_error = True
                        break

            # Only save sidecar JSON for successful results
            if not is_error and generated_text:
                try:
                    import re
                    data = {
                        "image_path": str(p.absolute()),
                        "image_filename": p.name,
                        "sglang": {
                            "full_output": generated_text
                        }
                    }
                    
                    match_r = re.search(r"<reasoning>([\s\S]*?)</reasoning>", generated_text)
                    match_a = re.search(r"<answer>([\s\S]*?)</answer>", generated_text)
                    if match_r: data["sglang"]["reasoning"] = match_r.group(1).strip()
                    if match_a: data["sglang"]["answer"] = match_a.group(1).strip()

                    json_path = p.with_suffix(".json")
                    with open(json_path, "w", encoding="utf-8") as jf:
                        json.dump(data, jf, indent=4, ensure_ascii=False)
                except Exception as e:
                    self.log.emit(f"Warning: Failed to write sidecar json for {p.name}: {e}")
            elif is_error:
                self.log.emit(f"[SGLang] Skipped {p.name} (error, will retry next run): {generated_text[:80]}")

            with self._progress_lock:
                self.images_done += 1
                if self.images_done % 5 == 0 or self.images_done == self.total_pending:
                    self.log.emit(f"Progress: {self.images_done}/{self.total_pending}")

            job_queue.task_done()

    def run(self):
        try:
            if not self.input_dir or not os.path.isdir(self.input_dir):
                self.log.emit(f"ERROR: input directory not found: {self.input_dir}")
                return
            if not self.output_jsonl_path:
                self.log.emit("ERROR: output JSONL path is empty.")
                return

            image_dir = Path(self.input_dir)
            self.log.emit(f"Scanning for {self.extension_glob} files...")
            all_image_paths = list(image_dir.rglob(self.extension_glob))

            already_processed = self._get_already_processed()
            pending_paths = [p for p in all_image_paths if p.name not in already_processed]
            if not pending_paths:
                self.log.emit("No new images to process.")
                return

            self.log.emit("Peeking image headers via imagesize for fast validation...")
            try:
                import imagesize
            except ImportError:
                imagesize = None

            valid_paths = []
            def peek_valid(p):
                try:
                    if imagesize is not None:
                        w, h = imagesize.get(str(p))
                        if w > 0 and h > 0:
                            return p
                    else:
                        if p.stat().st_size > 0:
                            return p
                except Exception:
                    pass
                return None

            import os
            import concurrent.futures
            workers = min(64, (os.cpu_count() or 4) * 4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                for res in pool.map(peek_valid, pending_paths):
                    if res:
                        valid_paths.append(res)

            if not valid_paths:
                self.log.emit("No valid images found after peek.")
                return

            concurrency = int(self.config.get("sglang_concurrency", 40))
            max_workers = max(1, min(concurrency, len(valid_paths)))
            self.log.emit(f"Processing {len(valid_paths)} images (concurrency={max_workers}, Pre-fetch=2 batches ahead)...")

            import queue
            import threading
            from PIL import Image
            import io
            import base64
            
            # Universal safety for large images
            Image.MAX_IMAGE_PIXELS = None
            
            job_queue = queue.Queue(maxsize=max_workers * 2)
            self.images_done = 0
            self.total_pending = len(valid_paths)
            self._progress_lock = threading.Lock()
            
            consumers = []
            for _ in range(max_workers):
                t = threading.Thread(target=self._worker_loop, args=(job_queue,))
                t.daemon = True
                t.start()
                consumers.append(t)
            
            for p in valid_paths:
                # Put the path directly so the threads can encode it in parallel
                job_queue.put(p)
                
            # Wait for all items to be processed
            for _ in range(max_workers):
                job_queue.put(None)
                
            for t in consumers:
                t.join()

        except Exception as e:
            self.log.emit(f"ERROR (sglang standalone): {e}")
            self.log.emit(traceback.format_exc())
