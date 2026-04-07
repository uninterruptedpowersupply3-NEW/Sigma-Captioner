import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import asyncio
import aiohttp
import base64
import json
import time
import os
from pathlib import Path
import threading
from PIL import Image
import io

# The Regex grammar to force the thought process and answer natively
REGEX_GRAMMAR = r"<reasoning>[\s\S]*?</reasoning>\s*<answer>[\s\S]*?</answer>"

class SGLangProGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SGLang 1M Image Auto-Captioner [PRO + REASONING]")
        self.root.geometry("900x850")
        self.root.configure(padx=20, pady=20)

        # Core Variables
        self.folder_path = tk.StringVar()
        self.output_file = tk.StringVar()
        # UPDATED: Defaulting to the OpenAI-compatible chat completions endpoint
        self.url = tk.StringVar(value="http://127.0.0.1:30000/v1/chat/completions")
        self.concurrency = tk.IntVar(value=40)
        self.extension = tk.StringVar(value="*.jpg")
        self.max_tokens = tk.IntVar(value=1024) 
        self.max_res = tk.IntVar(value=256) 
        
        # State Tracking
        self.is_running = False
        self.processed_count = 0
        self.total_images = 0
        self.start_time = 0
        
        self.pause_event = None 
        self.is_paused = False

        self.create_widgets()

    def create_widgets(self):
        # --- Configuration Frame ---
        cfg_frame = ttk.LabelFrame(self.root, text=" 1. Data Routing ", padding=10)
        cfg_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(cfg_frame, text="Input Images:").grid(row=0, column=0, sticky="e", pady=5)
        ttk.Entry(cfg_frame, textvariable=self.folder_path, width=60).grid(row=0, column=1, padx=5)
        ttk.Button(cfg_frame, text="Browse Dir", command=self.browse_folder).grid(row=0, column=2)

        ttk.Label(cfg_frame, text="Output JSONL:").grid(row=1, column=0, sticky="e", pady=5)
        ttk.Entry(cfg_frame, textvariable=self.output_file, width=60).grid(row=1, column=1, padx=5)
        ttk.Button(cfg_frame, text="Save As", command=self.browse_save).grid(row=1, column=2)

        # --- Context Per Expert Frame ---
        ctx_frame = ttk.LabelFrame(self.root, text=" 2. Expert Context & System Prompt ", padding=10)
        ctx_frame.pack(fill="x", pady=(0, 10))
        
        self.system_context = scrolledtext.ScrolledText(ctx_frame, height=4, font=("Consolas", 10))
        self.system_context.pack(fill="x")
        self.system_context.insert("1.0", 
            "You are a precise, data-extraction expert. Analyze the image and generate a technical test bank question. "
            "RULES: "
            "1. Output your reasoning inside <reasoning> tags. "
            "2. Output your final answer inside <answer> tags. "
            "3. Keep your reasoning to a maximum of 3 concise sentences."
        )

        # --- Engine Settings Frame ---
        net_frame = ttk.LabelFrame(self.root, text=" 3. Engine Settings ", padding=10)
        net_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(net_frame, text="SGLang URL:").grid(row=0, column=0, sticky="e", pady=5)
        ttk.Entry(net_frame, textvariable=self.url, width=40).grid(row=0, column=1, sticky="w", padx=5)

        opts_frame = ttk.Frame(net_frame)
        opts_frame.grid(row=1, column=0, columnspan=3, sticky="w", pady=5)
        
        ttk.Label(opts_frame, text="Max Concurrent:").pack(side="left")
        ttk.Entry(opts_frame, textvariable=self.concurrency, width=6).pack(side="left", padx=(5, 15))
        
        ttk.Label(opts_frame, text="Max Tokens/Expert:").pack(side="left")
        ttk.Entry(opts_frame, textvariable=self.max_tokens, width=6).pack(side="left", padx=(5, 15))

        ttk.Label(opts_frame, text="Max Image Res:").pack(side="left")
        ttk.Entry(opts_frame, textvariable=self.max_res, width=6).pack(side="left", padx=(5, 15))
        
        ttk.Label(opts_frame, text="Extension:").pack(side="left")
        ttk.Entry(opts_frame, textvariable=self.extension, width=6).pack(side="left", padx=5)

        # --- Controls & Progress ---
        ctrl_frame = ttk.Frame(self.root)
        ctrl_frame.pack(fill="x", pady=10)

        self.start_btn = ttk.Button(ctrl_frame, text="▶ START RUN", command=self.start_thread)
        self.start_btn.pack(side="left", padx=5, ipadx=10, ipady=5)
        
        self.pause_btn = ttk.Button(ctrl_frame, text="⏸ PAUSE", command=self.toggle_pause, state="disabled")
        self.pause_btn.pack(side="left", padx=5, ipadx=10, ipady=5)

        self.status_label = ttk.Label(ctrl_frame, text="Awaiting orders...", font=("Consolas", 10, "bold"))
        self.status_label.pack(side="right", padx=10)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=5)

        # --- Live Preview Matrix ---
        preview_frame = ttk.LabelFrame(self.root, text=" Live Output Preview ", padding=10)
        preview_frame.pack(fill="both", expand=True)

        self.preview_area = scrolledtext.ScrolledText(preview_frame, height=8, state='disabled', font=("Consolas", 10), bg="#1e1e1e", fg="#00ff00")
        self.preview_area.pack(fill="both", expand=True, pady=(0, 10))

        self.log_area = scrolledtext.ScrolledText(preview_frame, height=4, state='disabled', font=("Consolas", 8))
        self.log_area.pack(fill="x")

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder: self.folder_path.set(folder)

    def browse_save(self):
        file = filedialog.asksaveasfilename(defaultextension=".jsonl", filetypes=[("JSONL Files", "*.jsonl"), ("All Files", "*.*")])
        if file: self.output_file.set(file)

    def log(self, message):
        def _update():
            self.log_area.config(state='normal')
            self.log_area.insert(tk.END, message + "\n")
            self.log_area.see(tk.END)
            self.log_area.config(state='disabled')
        self.root.after(0, _update)

    def update_preview(self, json_str):
        def _update():
            self.preview_area.config(state='normal')
            self.preview_area.delete(1.0, tk.END)
            try:
                parsed = json.loads(json_str)
                formatted = json.dumps(parsed, indent=2)
                self.preview_area.insert(tk.END, formatted)
            except:
                self.preview_area.insert(tk.END, json_str)
            self.preview_area.config(state='disabled')
        self.root.after(0, _update)

    def update_status(self):
        if not self.is_running or self.total_images == 0: return
        elapsed = time.time() - self.start_time
        fps = self.processed_count / elapsed if elapsed > 0 else 0
        remaining = self.total_images - self.processed_count
        eta_secs = remaining / fps if fps > 0 else 0
        
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_secs))
        pct = (self.processed_count / self.total_images) * 100
        
        self.progress_var.set(pct)
        status_text = f"[{pct:.1f}%] {self.processed_count}/{self.total_images} | {fps:.1f} it/s | ETA: {eta_str}"
        self.status_label.config(text=status_text)

    def toggle_pause(self):
        if not self.is_running or not self.pause_event: return
        if self.is_paused:
            self.pause_event.set()
            self.is_paused = False
            self.pause_btn.config(text="⏸ PAUSE")
            self.log("▶ Resumed workflow.")
        else:
            self.pause_event.clear()
            self.is_paused = True
            self.pause_btn.config(text="▶ RESUME")
            self.log("⏸ Workflow paused. Active requests will finish, new ones will wait.")

    def start_thread(self):
        if not self.folder_path.get() or not self.output_file.get():
            messagebox.showerror("Config Error", "Please define both Input Directory and Output File.")
            return

        self.start_btn.config(state="disabled")
        self.pause_btn.config(state="normal")
        self.is_running = True
        self.is_paused = False
        
        self.log_area.config(state='normal')
        self.log_area.delete(1.0, tk.END)
        self.log_area.config(state='disabled')

        thread = threading.Thread(target=self.run_asyncio_loop, daemon=True)
        thread.start()

    def run_asyncio_loop(self):
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(self.main_async())

    def get_already_processed(self, filepath):
        processed = set()
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "file" in data: processed.add(data["file"])
                    except: pass
        return processed

    # --- THE OPTIMIZED DATA MANAGER LOGIC ---
    async def read_image_to_base64(self, file_path, target_res):
        def _read():
            with Image.open(file_path) as img:
                w, h = img.size
                mode = img.mode
                img_format = img.format

            if w <= target_res and h <= target_res and mode == 'RGB' and img_format == 'JPEG':
                with open(file_path, "rb") as f:
                    return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode('utf-8')

            with Image.open(file_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                if w > target_res or h > target_res:
                    img.thumbnail((target_res, target_res), Image.Resampling.LANCZOS)
                
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')
                
        return await asyncio.to_thread(_read)

    # --- UPDATED AI WORKER LOGIC ---
    async def process_image(self, session, sem, image_path, file_lock, sys_context, target_res):
        await self.pause_event.wait()
        async with sem:
            try:
                b64_img_uri = await self.read_image_to_base64(image_path, target_res)
                
                # Use OpenAI 'messages' format to trigger SGLang's multimodal Jinja template
                payload = {
                    "model": "default",
                    "messages": [
                        {"role": "system", "content": sys_context},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": b64_img_uri}},
                            {"type": "text", "text": "<image>\nPlease analyze this image based on the system instructions."}
                        ]}
                    ],
                    "response_format": {
                        "type": "regex",
                        "regex": REGEX_GRAMMAR
                    },
                    "max_tokens": self.max_tokens.get(),
                    "temperature": 0.7,     
                    "top_p": 0.8,           
                    "presence_penalty": 1.5
                }

                # Safety check: Force chat completions endpoint if the user left /generate in the GUI
                sglang_chat_url = self.url.get().replace("/generate", "/v1/chat/completions")

                async with session.post(sglang_chat_url, json=payload, timeout=90) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Parse the text out of the OpenAI-style response object
                        generated_text = result["choices"][0]["message"]["content"]
                        
                        # Strip markdown if the model leaks it despite the schema
                        if generated_text.startswith("```json"):
                            generated_text = generated_text.replace("```json\n", "", 1).replace("```", "")
                            
                        async with file_lock:
                            with open(self.output_file.get(), "a", encoding="utf-8") as f:
                                json_line = json.dumps({"file": image_path.name, "output": generated_text})
                                f.write(json_line + "\n")
                                f.flush() 
                        
                        self.update_preview(generated_text)
                    else:
                        err_text = await response.text()
                        self.log(f"HTTP {response.status} on {image_path.name}: {err_text[:100]}")

            except Exception as e:
                self.log(f"Error processing {image_path.name}: {str(e)}")
            finally:
                self.processed_count += 1
                if self.processed_count % 5 == 0: 
                    self.root.after(0, self.update_status)

    async def main_async(self):
        self.pause_event = asyncio.Event()
        self.pause_event.set()

        image_dir = Path(self.folder_path.get())
        out_file = self.output_file.get()
        ext = self.extension.get()
        sys_context = self.system_context.get("1.0", tk.END).strip()
        target_res = self.max_res.get()
        
        self.log(f"Scanning for {ext} files...")
        all_image_paths = list(image_dir.rglob(ext))
        
        already_processed_names = self.get_already_processed(out_file)
        if already_processed_names:
            self.log(f"Found {len(already_processed_names)} completed files. Skipping them...")
        
        pending_paths = [p for p in all_image_paths if p.name not in already_processed_names]
        self.total_images = len(pending_paths)
        self.processed_count = 0
        
        if self.total_images == 0:
            self.log("SUCCESS: No new images to process.")
            self.root.after(0, self.finish_run)
            return

        self.log(f"Booting {self.concurrency.get()} experts (Max Res: {target_res})...")
        self.start_time = time.time()
        
        sem = asyncio.Semaphore(self.concurrency.get())
        file_lock = asyncio.Lock()
        timeout = aiohttp.ClientTimeout(total=300)
        connector = aiohttp.TCPConnector(limit=self.concurrency.get() + 20, force_close=False)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [self.process_image(session, sem, path, file_lock, sys_context, target_res) for path in pending_paths]
            await asyncio.gather(*tasks)

        self.root.after(0, self.finish_run)

    def finish_run(self):
        self.is_running = False
        self.update_status()
        self.start_btn.config(state="normal")
        self.pause_btn.config(state="disabled", text="⏸ PAUSE")
        self.log(f"\nCompleted run.")
        messagebox.showinfo("Complete", "Operation finished successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SGLangProGUI(root)
    root.mainloop()