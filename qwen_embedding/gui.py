import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import asyncio
import os
from helper import AsyncBatcher, GPUImageProcessor
from backend import FastQwenEngine
import time
import sys
import traceback

class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)

    def flush(self):
        pass

class FastQwenGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fast Qwen3-VL Tagger")
        self.root.geometry("800x600")
        
        # Engine state
        self.image_processor = GPUImageProcessor()
        self.batcher = AsyncBatcher()
        self.batch_size = tk.IntVar(value=16)
        self.threshold = tk.DoubleVar(value=0.30)
        self.max_tags = tk.IntVar(value=50)
        
        # Performance Toggles
        self.precision_var = tk.StringVar(value="bf16")
        self.tf32_var = tk.BooleanVar(value=True)
        self.quant_var = tk.BooleanVar(value=False)
        self.compile_var = tk.BooleanVar(value=True)
        self.dynamic_compile_var = tk.BooleanVar(value=True)
        self.cuda_graphs_var = tk.BooleanVar(value=False)
        self.pinned_mem_var = tk.BooleanVar(value=True)
        self.prefetch_var = tk.IntVar(value=2)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Top Frame - Vocab
        frame_top = tk.Frame(self.root, pady=10)
        frame_top.pack(fill=tk.X)
        
        tk.Label(frame_top, text="Vocabulary Tags (comma separated):").pack(side=tk.TOP, anchor=tk.W, padx=10)
        self.vocab_entry = tk.Text(frame_top, height=4)
        self.vocab_entry.pack(side=tk.TOP, fill=tk.X, padx=10)
        self.vocab_entry.insert(tk.END, "masterpiece, best quality, scenery, 1girl, 1boy, car, animal, food, photorealistic, anime")
        
        btn_frame = tk.Frame(frame_top)
        btn_frame.pack(pady=5)
        
        tk.Button(btn_frame, text="1. Init & Precompute (Text Box)", command=self.init_engine, bg="lightblue").pack(side=tk.LEFT, padx=5)
        
        cache_frame = tk.Frame(frame_top)
        cache_frame.pack(pady=5)
        tk.Label(cache_frame, text="Inductor/Vocab JSON Cache Path:").pack(side=tk.LEFT, padx=5)
        self.cache_path_var = tk.StringVar(value=os.path.join(os.getcwd(), "vocab_hybrid_meta.json"))
        tk.Entry(cache_frame, textvariable=self.cache_path_var, width=50).pack(side=tk.LEFT, padx=5)
        
        tk.Button(cache_frame, text="1b. Load Selected JSON Cache File", command=self.load_json_cache_from_entry, bg="lightyellow").pack(side=tk.LEFT, padx=5)
        
        # Middle Frame - Image Selection
        frame_mid = tk.Frame(self.root, pady=10)
        frame_mid.pack(fill=tk.X)
        
        self.folder_label = tk.Label(frame_mid, text="No folder selected")
        self.folder_label.pack(side=tk.TOP)
        tk.Button(frame_mid, text="2. Select Data Folder (Images or .pt Embeddings)", command=self.select_folder).pack()
        
        # Batch Size Parameter
        frame_bs = tk.Frame(frame_mid)
        frame_bs.pack(pady=10)
        tk.Label(frame_bs, text="GPU Image Batch Size:").pack(side=tk.LEFT)
        tk.Entry(frame_bs, textvariable=self.batch_size, width=8).pack(side=tk.LEFT, padx=5)
        
        tk.Label(frame_bs, text="Confidence Threshold:").pack(side=tk.LEFT, padx=(15, 5))
        tk.Entry(frame_bs, textvariable=self.threshold, width=6).pack(side=tk.LEFT)
        
        tk.Label(frame_bs, text="Max Tags:").pack(side=tk.LEFT, padx=(15, 5))
        tk.Entry(frame_bs, textvariable=self.max_tags, width=5).pack(side=tk.LEFT)
        
        frame_perf1 = tk.Frame(frame_mid)
        frame_perf1.pack(pady=3)
        tk.Label(frame_perf1, text="Precision:").pack(side=tk.LEFT)
        tk.OptionMenu(frame_perf1, self.precision_var, "bf16", "fp32", "int8").pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(frame_perf1, text="TF32 Math", variable=self.tf32_var).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(frame_perf1, text="AWQ/4-bit Quant", variable=self.quant_var).pack(side=tk.LEFT, padx=5)
        frame_perf2 = tk.Frame(frame_mid)
        frame_perf2.pack(pady=3)
        tk.Checkbutton(frame_perf2, text="torch.compile", variable=self.compile_var).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(frame_perf2, text="Dynamic Shapes", variable=self.dynamic_compile_var).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(frame_perf2, text="CUDA Graphs", variable=self.cuda_graphs_var).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(frame_perf2, text="Pinned Memory", variable=self.pinned_mem_var).pack(side=tk.LEFT, padx=5)
        tk.Label(frame_perf2, text="Prefetch:").pack(side=tk.LEFT, padx=5)
        tk.Entry(frame_perf2, textvariable=self.prefetch_var, width=3).pack(side=tk.LEFT)
        
        self.selected_folder = ""
        
        # Action Frame
        frame_action = tk.Frame(self.root, pady=10)
        frame_action.pack(fill=tk.X)
        
        tk.Button(frame_action, text="3. Run Inference (Max Throughput)", command=self.start_inference, bg="lightgreen").pack()
        
        # Log frame
        self.log_text = tk.Text(self.root, height=10, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Global Redirection: Capture EVERYTHING from sys.stdout/stderr (including Torch/Inductor logs)
        redir = StdoutRedirector(self.log_text)
        sys.stdout = redir
        sys.stderr = redir
        
    def log(self, message):
        def _update_ui():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        self.root.after(0, _update_ui)
        
    def init_engine(self):
        def _init():
            try:
                self.log("Initializing FastQwenEngine. This may take a minute...")
                self.engine = FastQwenEngine(
                    precision=self.precision_var.get(),
                    tf32=self.tf32_var.get(),
                    quant=self.quant_var.get()
                )
                
                vocab = [v.strip() for v in self.vocab_entry.get("1.0", tk.END).split(",") if v.strip()]
                self.log(f"Precomputing cache for {len(vocab)} tags...")
                self.engine.prepare_vocabulary(vocab)
                self.log("Engine Ready.")
            except Exception as e:
                self.log(f"Error init engine: {e}")
                
        threading.Thread(target=_init, daemon=True).start()
        
    def load_json_cache(self):
        json_file = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if not json_file:
            return
            
        def _load():
            try:
                self.log("Initializing FastQwenEngine. This may take a minute...")
                self.engine = FastQwenEngine(
                    precision=self.precision_var.get(),
                    tf32=self.tf32_var.get(),
                    quant=self.quant_var.get()
                )
                
                self.log(f"Loading massive vocabulary cache from {json_file}...")
                self.engine.load_vocab_cache(json_file)
                self.log("Engine Ready with JSON cache.")
            except Exception as e:
                self.log(f"Error loading JSON cache engine: {e}")
                
        threading.Thread(target=_load, daemon=True).start()

    def load_json_cache_from_entry(self):
        json_file = self.cache_path_var.get()
        if not json_file or not os.path.exists(json_file):
            messagebox.showerror("Error", "Selected file does not exist!")
            return
            
        def _load():
            try:
                self.log("Initializing FastQwenEngine. This may take a minute...")
                self.engine = FastQwenEngine(
                    precision=self.precision_var.get(),
                    tf32=self.tf32_var.get(),
                    quant=self.quant_var.get()
                )
                
                self.log(f"Loading massive vocabulary cache from {json_file}...")
                self.engine.load_vocab_cache(json_file)
                self.log("Engine Ready with JSON cache.")
            except Exception as e:
                self.log(f"Error loading JSON cache engine: {e}")
                
        threading.Thread(target=_load, daemon=True).start()
        
    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.selected_folder = folder
            self.folder_label.config(text=folder)
            
    def start_inference(self):
        if not self.engine:
            messagebox.showerror("Error", "Initialize engine first!")
            return
        if not self.selected_folder:
            messagebox.showerror("Error", "Select folder first!")
            return
            
        def _run():
            self.log("Scanning directory...")
            img_files = [os.path.join(self.selected_folder, f) for f in os.listdir(self.selected_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            pt_files = [os.path.join(self.selected_folder, f) for f in os.listdir(self.selected_folder) 
                        if f.lower().endswith('.pt')]

            packed_cache_path = os.path.join(self.selected_folder, "qwen_image_cache.pt")
            if os.path.isfile(packed_cache_path):
                self.log(f"Found packed image cache ({os.path.basename(packed_cache_path)}). Running cached inference...")
                import torch
                cache = torch.load(packed_cache_path, map_location="cpu")
                embeddings = cache.get("embeddings")
                paths = cache.get("paths")

                if embeddings is None or paths is None:
                    self.log("ERROR: packed cache is missing 'embeddings' or 'paths'.")
                    return
                if not hasattr(embeddings, "dim") or embeddings.dim() != 2:
                    self.log(f"ERROR: packed cache embeddings must be 2D [N, D]. Got: {getattr(embeddings, 'shape', None)}")
                    return
                if len(paths) != int(embeddings.shape[0]):
                    self.log(f"ERROR: packed cache paths count ({len(paths)}) != embeddings N ({int(embeddings.shape[0])}).")
                    return

                total_processed = 0
                t0 = time.time()
                write_buffer = []
                chunk_size = max(1024, self.batch_size.get() * 64)
                for start in range(0, int(embeddings.shape[0]), chunk_size):
                    end = min(start + chunk_size, int(embeddings.shape[0]))
                    batch_latents = embeddings[start:end]
                    results = self.engine.predict_from_embeddings(
                        batch_latents,
                        threshold=self.threshold.get(),
                        max_tags=self.max_tags.get(),
                    )

                    total_processed += len(results)
                    for rel_path, res in zip(paths[start:end], results):
                        img_path = rel_path
                        if not os.path.isabs(img_path):
                            img_path = os.path.join(self.selected_folder, img_path)
                        clean_tags = ", ".join([t for t, _s in res])
                        txt_path = os.path.splitext(img_path)[0] + ".txt"
                        write_buffer.append((txt_path, clean_tags))

                    if len(write_buffer) >= 1000:
                        buf = write_buffer.copy()
                        write_buffer.clear()
                        def _flush(b):
                            for p, t in b:
                                try:
                                    with open(p, 'w', encoding='utf-8') as f:
                                        f.write(t)
                                except Exception:
                                    pass
                        threading.Thread(target=_flush, args=(buf,), daemon=True).start()

                    self.log(f"Processed {total_processed}/{int(embeddings.shape[0])} cached embeddings...")

                if write_buffer:
                    for p, t in write_buffer:
                        try:
                            with open(p, 'w', encoding='utf-8') as f:
                                f.write(t)
                        except Exception:
                            pass

                t1 = time.time()
                self.log(f"Done! Evaluated {total_processed} cached embeddings in {t1-t0:.2f} seconds.")
                return
            
            if len(pt_files) > len(img_files) and len(pt_files) > 0:
                self.log(f"Found {len(pt_files)} embedding (.pt) files! Running ultra-fast tensor inference...")
                import torch
                total_processed = 0
                t0 = time.time()
                write_buffer = []
                
                # CPU Matmul is instantaneous, process in massive chunks
                chunk_size = max(64, self.batch_size.get() * 4) 
                chunks = [pt_files[i:i+chunk_size] for i in range(0, len(pt_files), chunk_size)]
                
                for chunk in chunks:
                    tensors = []
                    for pt in chunk:
                        tensors.append(torch.load(pt, map_location="cpu"))
                    
                    batched_latents = torch.cat(tensors, dim=0)
                    results = self.engine.predict_from_embeddings(
                        batched_latents,
                        threshold=self.threshold.get(),
                        max_tags=self.max_tags.get()
                    )
                    
                    total_processed += len(results)
                    for path, res in zip(chunk, results):
                        log_tags = ", ".join([f"{t}({s:.2f})" for t, s in res])
                        if total_processed % chunk_size == 0 or len(chunk) < chunk_size:
                            # Log periodically to prevent GUI freeze
                            self.log(f"{os.path.basename(path)}: {log_tags}")
                        
                        clean_tags = ", ".join([t for t, s in res])
                        txt_path = os.path.splitext(path)[0] + ".txt"
                        write_buffer.append((txt_path, clean_tags))
                        
                    if len(write_buffer) >= 1000:
                        buf = write_buffer.copy()
                        write_buffer.clear()
                        def _flush(b):
                            for p, t in b:
                                try:
                                    with open(p, 'w', encoding='utf-8') as f:
                                        f.write(t)
                                except Exception:
                                    pass
                        threading.Thread(target=_flush, args=(buf,), daemon=True).start()
                
                if write_buffer:
                    for p, t in write_buffer:
                        try:
                            with open(p, 'w', encoding='utf-8') as f:
                                f.write(t)
                        except Exception:
                            pass
                
                t1 = time.time()
                self.log(f"Done! Evaluated {total_processed} embeddings against vocab in {t1-t0:.2f} seconds.")
                self.log(f"Tensor Throughput: {total_processed / max(1e-5, (t1-t0)):.2f} latents/s")
                return
                
            files = img_files
            self.log(f"Found {len(files)} images. Grouping via Peek Resizing...")
            
            from concurrent.futures import ThreadPoolExecutor
            
            t0 = time.time()
            groups = self.image_processor.peek_and_group(files)
            self.log(f"Grouped into {len(groups)} uniform resolution batches.")
            
            total_processed = 0
            write_buffer = []
            max_worker_pool = max(1, self.prefetch_var.get())
            
            with ThreadPoolExecutor(max_workers=max_worker_pool) as executor:
                for target_size, paths in groups.items():
                    self.log(f"Processing batch shape {target_size} ({len(paths)} images) with {max_worker_pool}x CPU prefetch overlaps...")
                    chunk_size = self.batch_size.get()
                    
                    chunks = [paths[i:i+chunk_size] for i in range(0, len(paths), chunk_size)]
                    futures = []
                    
                    def handle_results(p_result):
                        if not p_result:
                            return
                        v_paths, p_images = p_result
                        if not p_images:
                            return
                            
                        nonlocal total_processed, write_buffer
                        results = self.engine.predict_batch(
                            p_images, 
                            threshold=self.threshold.get(), 
                            max_tags=self.max_tags.get(),
                            use_compile=self.compile_var.get(),
                            dynamic_compile=self.dynamic_compile_var.get(), # Added this line
                            use_cuda_graphs=self.cuda_graphs_var.get(),
                            use_pinned_memory=self.pinned_mem_var.get()
                        )
                        total_processed += len(results)
                        for path, res in zip(v_paths, results):
                            log_tags = ", ".join([f"{t}({s:.2f})" for t, s in res])
                            self.log(f"{os.path.basename(path)}: {log_tags}")
                            clean_tags = ", ".join([t for t, s in res])
                            txt_path = os.path.splitext(path)[0] + ".txt"
                            write_buffer.append((txt_path, clean_tags))
                            
                        if len(write_buffer) >= 1000:
                            def _flush(buf):
                                try:
                                    for p, t in buf:
                                        with open(p, 'w', encoding='utf-8') as f:
                                            f.write(t)
                                except Exception as e:
                                    print(f"Batch write failed: {e}")
                            threading.Thread(target=_flush, args=(write_buffer.copy(),), daemon=True).start()
                            write_buffer.clear()
                            
                    for chunk in chunks:
                        futures.append((chunk, executor.submit(self.image_processor.process_group, chunk, target_size)))
                        if len(futures) >= max_worker_pool:
                            _, p_future = futures.pop(0)
                            handle_results(p_future.result())
                            
                    for _, p_future in futures:
                        handle_results(p_future.result())
                        
            # Final flush for any remaining data
            if write_buffer:
                for p, t in write_buffer:
                    try:
                        with open(p, 'w', encoding='utf-8') as f:
                            f.write(t)
                    except Exception:
                        pass
                write_buffer.clear()
                        
            t1 = time.time()
            self.log(f"Done! Processed {total_processed} images in {t1-t0:.2f} seconds.")
            self.log(f"Throughput: {total_processed / max(1e-5, (t1-t0)):.2f} img/s")
            
        def _run_safe():
            try:
                _run()
            except Exception:
                traceback.print_exc()
                
        threading.Thread(target=_run_safe, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = FastQwenGUI(root)
    root.mainloop()
