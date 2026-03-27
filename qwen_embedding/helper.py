import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Any
from PIL import Image
import os
try:
    import imagesize
except ImportError:
    imagesize = None

class GPUImageProcessor:
    """
    Extremely optimized Windows GPU-based image resizing and processing.
    Uses 'peek resizing' via imagesize to calculate sizes without full loading.
    """
    def __init__(self, device="cuda"):
        self.device = device
        
    def _peek_single(self, args) -> Tuple[str, int, int]:
        path, max_pixels = args
        w, h = None, None
        if imagesize:
            try:
                w, h = imagesize.get(path)
            except Exception:
                pass
        
        # Fallback to PIL if imagesize fails or is missing
        if not w or not h:
            try:
                with Image.open(path) as img:
                    w, h = img.size
            except Exception:
                return path, None, None
                
        total_pixels = w * h
        if total_pixels <= max_pixels:
            target_w, target_h = w, h
        else:
            scale = (max_pixels / total_pixels) ** 0.5
            target_w, target_h = int(w * scale), int(h * scale)
        
        # Resolution Quantization: Snap to 28-pixel grid (Qwen patch size)
        target_w = (target_w // 28) * 28
        target_h = (target_h // 28) * 28
        
        # Ensure at least one patch
        target_w = max(target_w, 28)
        target_h = max(target_h, 28)
            
        return path, target_w, target_h

    def peek_and_group(self, file_paths: List[str], max_pixels: int = 250880, progress_callback=None, log_callback=None) -> Dict[Tuple[int, int], List[str]]:
        """
        Groups images by their optimal destination size using fast header peeking via ThreadPoolExecutor.
        """
        groups = {}
        total = len(file_paths)
        args_list = [(p, max_pixels) for p in file_paths]
        
        # Utilize maximum standard CPU threads dedicated for concurrent disk I/O
        workers = min(64, (os.cpu_count() or 4) * 4)
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for i, result in enumerate(pool.map(self._peek_single, args_list)):
                path, target_w, target_h = result
                if target_w is not None and target_h is not None:
                    groups.setdefault((target_w, target_h), []).append(path)
                else:
                    if log_callback:
                        log_callback(f"⚠️ Skipped unreadable/corrupted image: {path}")
                
                if progress_callback and (i % 50 == 0 or i == total - 1):
                    progress_callback(i + 1, total, f"Scanning dimensions ({i + 1}/{total})...")
                    
        return groups
        
    def process_group(self, group_paths: List[str], target_size: Tuple[int, int]) -> Tuple[List[str], List[Image.Image]]:
        """
        Process a group of images that share the identical target size padding efficiently on GPU.
        """
        processed_pil = []
        valid_paths = []
        target_w, target_h = target_size
        
        with torch.no_grad():
            for path in group_paths:
                try:
                    img = Image.open(path)
                    
                    # If already perfectly sized, skip eager decoding until the engine processor needs it.
                    if img.size == target_size:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        processed_pil.append(img)
                        valid_paths.append(path)
                        continue
                        
                    # If it needs resizing, we must decode eagerly anyway
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    
                    t = TF.to_tensor(img).to(self.device).unsqueeze(0)
                    t_resized = F.interpolate(t, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    final_img = TF.to_pil_image(t_resized.squeeze(0).cpu())
                    processed_pil.append(final_img)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    
        return valid_paths, processed_pil

class AsyncBatcher:
    """
    Optimized batch processor using multithreading for I/O and CPU bound ops 
    to not lock the main Tkinter thread.
    """
    def __init__(self, max_workers: int = 8):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def process_in_batches(self, items: List[Any], batch_size: int, func):
        loop = asyncio.get_event_loop()
        tasks = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            task = loop.run_in_executor(self.executor, func, batch)
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]
