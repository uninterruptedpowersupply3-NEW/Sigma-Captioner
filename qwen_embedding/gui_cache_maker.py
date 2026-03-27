import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import sys
import os
import time
import torch
from vocab_generator import VocabCacheGenerator
from backend import FastQwenEngine
from helper import GPUImageProcessor
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

class PreprocessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Qwen3-VL Preprocessing Workstation")
        self.root.geometry("800x700")
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.tab_vocab = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_vocab, text="📚 Build Vocab Cache")
        
        self.tab_latents = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_latents, text="🖼️ Extract Image Latents")
        
        # Vocab Variables
        self.danbooru_path = tk.StringVar(value=os.path.join(os.getcwd(), "data.json"))
        self.min_occurrences = tk.IntVar(value=200)
        self.vocab_bs = tk.IntVar(value=64)
        self.use_english = tk.BooleanVar(value=True)
        self.output_pt_path = tk.StringVar(value=os.path.join(os.getcwd(), "vocab_hybrid_matrix.pt"))
        self.output_json_path = tk.StringVar(value=os.path.join(os.getcwd(), "vocab_hybrid_meta.json"))
        
        # Latent Variables
        self.image_folder = tk.StringVar(value="")
        self.precision = tk.StringVar(value="bf16")
        self.tf32_var = tk.BooleanVar(value=True)
        self.compile_var = tk.BooleanVar(value=True)
        self.dynamic_compile_var = tk.BooleanVar(value=True)
        self.pinned_mem_var = tk.BooleanVar(value=True)
        self.cuda_graphs_var = tk.BooleanVar(value=False)
        self.prefetch_var = tk.IntVar(value=2)
        self.latent_bs = tk.IntVar(value=8)

        self.btn_vocab: tk.Button = None
        self.btn_latents: tk.Button = None
        self.log_text: tk.Text = None
        
        self.setup_vocab_ui()
        self.setup_latents_ui()
        self.setup_logger()
        
    def setup_vocab_ui(self):
        f = tk.LabelFrame(self.tab_vocab, text="Danbooru JSON Source", padx=10, pady=10)
        f.pack(fill=tk.X, padx=10, pady=5)
        tk.Entry(f, textvariable=self.danbooru_path, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(f, text="Browse", command=lambda: self.danbooru_path.set(filedialog.askopenfilename(filetypes=[("JSON", "*.json")]))).pack(side=tk.LEFT, padx=5)
        
        s = tk.LabelFrame(self.tab_vocab, text="Settings", padx=10, pady=10)
        s.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(s, text="Min Occurrences:").grid(row=0, column=0, sticky=tk.W)
        tk.Entry(s, textvariable=self.min_occurrences, width=10).grid(row=0, column=1, padx=5)
        tk.Label(s, text="Batch Size (GPU):").grid(row=1, column=0, sticky=tk.W)
        tk.Entry(s, textvariable=self.vocab_bs, width=10).grid(row=1, column=1, padx=5)
        tk.Checkbutton(s, text="Merge 370k DWYL English Dictionary", variable=self.use_english).grid(row=2, column=0, columnspan=2, sticky=tk.W)
        
        tk.Label(s, text="Output Matrix (.pt):").grid(row=3, column=0, sticky=tk.W)
        tk.Entry(s, textvariable=self.output_pt_path, width=40).grid(row=3, column=1, padx=5, sticky=tk.W)
        tk.Label(s, text="Output Meta (.json):").grid(row=4, column=0, sticky=tk.W)
        tk.Entry(s, textvariable=self.output_json_path, width=40).grid(row=4, column=1, padx=5, sticky=tk.W)
        
        self.btn_vocab = tk.Button(self.tab_vocab, text="⚡ Generate Massive GPU Cache (.pt / .json)", bg="lightgreen", command=self.run_vocab)
        self.btn_vocab.pack(fill=tk.X, padx=10, pady=10)
        
    def setup_latents_ui(self):
        f = tk.LabelFrame(self.tab_latents, text="Image Directory", padx=10, pady=10)
        f.pack(fill=tk.X, padx=10, pady=5)
        tk.Entry(f, textvariable=self.image_folder, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(f, text="Browse", command=lambda: self.image_folder.set(filedialog.askdirectory())).pack(side=tk.LEFT, padx=5)
        
        p = tk.LabelFrame(self.tab_latents, text="Pipeline Performance Specs", padx=10, pady=10)
        p.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(p, text="Precision:").grid(row=0, column=0, sticky=tk.E)
        ttk.Combobox(p, textvariable=self.precision, values=["bf16", "fp32", "int8", "int4"], width=8).grid(row=0, column=1, sticky=tk.W)
        
        tk.Label(p, text="Batch Size:").grid(row=0, column=2, sticky=tk.E)
        tk.Entry(p, textvariable=self.latent_bs, width=6).grid(row=0, column=3, sticky=tk.W)
        
        # Performance Checkboxes
        perf_frame = tk.Frame(p)
        perf_frame.grid(row=1, column=0, columnspan=5, sticky=tk.W)
        tk.Checkbutton(perf_frame, text="torch.compile", variable=self.compile_var).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(perf_frame, text="Dynamic Shapes", variable=self.dynamic_compile_var).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(perf_frame, text="CUDA Graphs", variable=self.cuda_graphs_var).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(perf_frame, text="Pinned Memory", variable=self.pinned_mem_var).pack(side=tk.LEFT, padx=5)
        
        self.btn_latents = tk.Button(self.tab_latents, text="🧠 Extract Image Latents (.pt)", bg="lightblue", command=self.run_latents)
        self.btn_latents.pack(fill=tk.X, padx=10, pady=10)
        
    def setup_logger(self):
        frame_log = tk.LabelFrame(self.root, text="Execution Log", padx=10, pady=5)
        frame_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(frame_log, state=tk.DISABLED, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        sys.stdout = StdoutRedirector(self.log_text)
        sys.stderr = sys.stdout

    def run_vocab(self):
        self.btn_vocab.config(state=tk.DISABLED)
        def _task():
            try:
                print("Starting Hybrid Vocab Extraction...")
                gen = VocabCacheGenerator(precision="int8")
                gen.build_hybrid_danbooru_english_vocab(
                    self.danbooru_path.get(),
                    self.output_pt_path.get(),
                    self.output_json_path.get(),
                    self.min_occurrences.get(),
                    self.use_english.get(),
                    self.vocab_bs.get()
                )
                print("✅ Vocab Generation Complete!")
            except Exception as e:
                print(f"❌ ERROR: {e}")
                traceback.print_exc()
            finally:
                self.btn_vocab.config(state=tk.NORMAL)
        threading.Thread(target=_task, daemon=True).start()

    def run_latents(self):
        if not self.image_folder.get():
            messagebox.showerror("Error", "Select image directory first!")
            return
            
        self.btn_latents.config(state=tk.DISABLED)
        
        def _task():
            try:
                print(f"Initializing FastQwenEngine ({self.precision.get()})...")
                engine = FastQwenEngine(precision=self.precision.get(), tf32=self.tf32_var.get(), quant=self.precision.get() in ["int8", "int4"])
                processor = GPUImageProcessor()
                
                folder = self.image_folder.get()
                files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                print(f"Scanning {len(files)} images for dynamic grouping...")
                t0 = time.time()
                groups = processor.peek_and_group(files)
                print(f"Grouped into {len(groups)} uniform resolution blocks.")
                
                from concurrent.futures import ThreadPoolExecutor
                total = [0]
                embedding_chunks = []
                all_rel_paths = []
                threads = self.prefetch_var.get()
                
                with ThreadPoolExecutor(max_workers=max(1, threads)) as executor:
                    for target_size, paths in groups.items():
                        print(f"Batch mapping shape {target_size} ({len(paths)} images)...")
                        bs = self.latent_bs.get()
                        chunks = [paths[i:i+bs] for i in range(0, len(paths), bs)]
                        futures = []
                        
                        def write_tensors(chunk_paths, p_images):
                            if not p_images:
                                return
                            # Direct Latent Subroutine Mapping
                            tensor_outputs = engine.extract_image_features(
                                p_images, 
                                use_compile=self.compile_var.get(),
                                dynamic_compile=self.dynamic_compile_var.get(),
                                use_cuda_graphs=self.cuda_graphs_var.get(),
                                use_pinned_memory=self.pinned_mem_var.get()
                            )
                            embedding_chunks.append(tensor_outputs.clone().detach().cpu())
                            all_rel_paths.extend([os.path.relpath(p, folder) for p in chunk_paths])
                            total[0] += len(chunk_paths)
                            
                        for chunk in chunks:
                            futures.append((chunk, executor.submit(processor.process_group, chunk, target_size)))
                            if len(futures) >= threads:
                                _, fut = futures.pop(0)
                                v_paths, p_imgs = fut.result()
                                write_tensors(v_paths, p_imgs)
                                
                        for _, fut in futures:
                            v_paths, p_imgs = fut.result()
                            write_tensors(v_paths, p_imgs)
                            
                if embedding_chunks and all_rel_paths:
                    packed = torch.cat(embedding_chunks, dim=0).contiguous()
                    out_path = os.path.join(folder, "qwen_image_cache.pt")
                    torch.save({"paths": all_rel_paths, "embeddings": packed}, out_path)
                    print(f"\n✅ Transcoded {total[0]} raw latents in {time.time()-t0:.2f} seconds!")
                    print(f"💾 Saved packed cache: {out_path}  (N={packed.shape[0]}, D={packed.shape[1]})")
                else:
                    print("⚠️ No embeddings generated (all images may have failed to load).")
                
            except Exception as e:
                import traceback
                print(f"❌ ERROR: {e}")
                traceback.print_exc()
            finally:
                self.btn_latents.config(state=tk.NORMAL)
                
        threading.Thread(target=_task, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = PreprocessingGUI(root)
    root.mainloop()
