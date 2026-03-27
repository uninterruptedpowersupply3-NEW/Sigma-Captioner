import os
import time
import torch
from PIL import Image
import tempfile
from backend import FastQwenEngine
from helper import GPUImageProcessor

# Ensure offline loading
os.environ["HF_HUB_OFFLINE"] = "1"

def run_benchmark(device_type):
    print(f"\n--- Starting Benchmark: {device_type.upper()} ---")
    
    # Initialize engine
    # For CPU, we cannot rely on bf16/tf32/flash_attention
    if device_type == "cpu":
        engine = FastQwenEngine(
            precision="fp32", 
            device="cpu", 
            tf32=False, 
            quant=False
        )
    else:
        # GPU Max Performance Settings
        engine = FastQwenEngine(
            precision="bf16", 
            device="cuda", 
            tf32=True, 
            quant=False
        )
        
    vocab = ["masterpiece", "best quality", "scenery", "1girl", "1boy", "car", "animal"]
    engine.prepare_vocabulary(vocab)
    
    # Create fake batch of 16 images
    print("Generating synthetic 512x512 image batch...")
    img = Image.new('RGB', (512, 512), color='blue')
    tmp_path = "temp_bench_image.jpg"
    img.save(tmp_path, format="JPEG")
        
    processor = GPUImageProcessor()
    target_size = (512, 512) # Uniform
    
    # GPU processor outputs a batch of 16 identical inputs
    processed_images = processor.process_group([tmp_path] * 16, target_size)
    try:
        os.remove(tmp_path)
    except:
        pass
    
    print("Warmup Pass...")
    use_compile = True if device_type == "cuda" else False
    use_cuda_graphs = False
    
    try:
        # 1 Warmup pass
        engine.predict_batch(
            processed_images, 
            use_compile=use_compile, 
            use_cuda_graphs=use_cuda_graphs, 
            use_pinned_memory=False
        )
    except Exception as e:
        print(f"Warmup failed: {e}. Disabling graphs/compile and retrying...")
        use_compile = False
        use_cuda_graphs = False
        engine.predict_batch(
            processed_images, 
            use_compile=False, 
            use_cuda_graphs=False, 
            use_pinned_memory=False
        )
        
    batches = 10 if device_type == "cuda" else 2
    print(f"Benchmarking {batches} batches ({batches * 16} total images)...")
    if device_type == "cuda":
        torch.cuda.synchronize()
        
    t0 = time.time()
    for _ in range(batches):
        engine.predict_batch(
            processed_images, 
            use_compile=use_compile, 
            use_cuda_graphs=use_cuda_graphs, 
            use_pinned_memory=False
        )
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    
    elapsed = t1 - t0
    total_imgs = batches * 16
    print(f"\n=== RESULTS ({device_type.upper()}) ===")
    print(f"Total Images: {total_imgs}")
    print(f"Time Elapsed: {elapsed:.2f} seconds")
    print(f"Throughput: {total_imgs / elapsed:.2f} images/sec")
    print("===================\n")

import sys

if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    try:
        run_benchmark(device)
    except Exception as e:
        print(f"\n[!] {device.upper()} Benchmark completely failed: {e}\n")
