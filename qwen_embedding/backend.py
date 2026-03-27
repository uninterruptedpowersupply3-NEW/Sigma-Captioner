import os
import gc
import json
import torch
import torch.nn.functional as F
import torch._dynamo as dynamo
import sys
from typing import List, Tuple, Optional
from PIL import Image

class FastQwenEngine:
    def __init__(
        self,
        model_id: Optional[str] = None,
        precision: str = "bf16",
        tf32: bool = True,
        quant: bool = False,
        inductor_cache_dir: Optional[str] = None,
        inductor_compile_threads: int = 0,
        device: Optional[str] = None,
        output_dim: Optional[int] = None,
        log_callback=None
    ):
        self.model_id = model_id
        if not self.model_id:
            # Fallback 1: Local Sigma-Captioner models folder
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            standard_qwen = os.path.join(base_dir, "models", "Qwen")
            if os.path.isdir(standard_qwen):
                self.model_id = standard_qwen
            else:
                # Fallback 2: Hugging Face default ID (for online/cache loading)
                self.model_id = "Qwen/Qwen3.5-2B"
        self.precision = precision
        self.tf32 = tf32
        self.quant = quant
        self.log_callback = log_callback or print
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Add script path to sys.path for importing the official embedder
        script_dirs = []
        if self.model_id and os.path.isdir(self.model_id):
            script_dirs.append(os.path.join(self.model_id, "scripts"))
        
        # Fallback to standard locations relative to the engine
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Sigma-Captioner root
        script_dirs.append(os.path.join(base_dir, "models", "Qwen", "scripts"))
        
        found_any = False
        for sdir in script_dirs:
            if os.path.isdir(sdir):
                if sdir not in sys.path:
                    sys.path.insert(0, sdir)
                self.log_callback(f"Added model scripts to path: {sdir}")
                found_any = True
                break
        
        if not found_any:
            self.log_callback("Warning: Could not find any Qwen script directories.")

        try:
            from qwen3_vl_embedding import Qwen3VLEmbedder
            self.log_callback("Successfully imported official Qwen3VLEmbedder from model scripts.")
        except ImportError as e:
            self.log_callback(f"Error: Could not import Qwen3VLEmbedder: {e}")
            raise ImportError("Failed to locate Qwen3VLEmbedder in model scripts.")

        # Map precision to dtype AND kwargs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16
        kwargs = {}
        if precision == "int8":
            kwargs["load_in_8bit"] = True
            self.dtype = torch.float16 # Weights are 8bit, activations usually 16bit
        elif precision == "int4":
            kwargs["load_in_4bit"] = True
            self.dtype = torch.float16
        elif precision == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16
            self.dtype = torch.bfloat16
        elif precision == "fp32":
             kwargs["torch_dtype"] = torch.float32
             self.dtype = torch.float32
        else:
            kwargs["torch_dtype"] = torch.float16
            self.dtype = torch.float16

        if tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.log_callback(f"Initializing Qwen3VLEmbedder with precision: {precision}...")
        
        # Determine attention implementation for loading
        # Standard transformers keyword: "sdpa" (Scaled Dot Product Attention)
        # Optimized for Ampere/Ada GPUs (RTX 30/40) and fits in ~4GB VRAM
        load_attn_impl = "sdpa"

        self.embedder = Qwen3VLEmbedder(
            model_name_or_path=self.model_id,
            attn_implementation=load_attn_impl,
            min_pixels=256 * 256, # Optimized for 256x256 ImageNet
            max_pixels=256 * 256, 
            **kwargs
        )

        self.log_callback(f"Qwen3VLEmbedder initialized with {load_attn_impl} attention.")

        self.log_callback("Qwen3VLEmbedder initialized.")

        # Inject Chat Template if missing (cures local model folders without tokenizer_config.json)
        if hasattr(self.embedder, "processor"):
            p = self.embedder.processor
            if not hasattr(p, "chat_template") or p.chat_template is None:
                self.log_callback("Notice: Injecting default Chat Template into Qwen processor...")
                p.chat_template = (
                    "{% for message in messages %}"
                    "{{ '<|im_start|>' + message['role'] + '\\n' }}"
                    "{% if message['content'] is string %}"
                    "{{ message['content'] }}"
                    "{% else %}"
                    "{% for content in message['content'] %}"
                    "{% if content['type'] == 'text' %}{{ content['text'] }}"
                    "{% elif content['type'] == 'image' %}{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
                    "{% elif content['type'] == 'video' %}{{ '<|vision_start|><|video_pad|><|vision_end|>' }}"
                    "{% endif %}"
                    "{% endfor %}"
                    "{% endif %}"
                    "{{ '<|im_end|>\\n' }}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
                )

        self.vocab_words = []
        self.vocab_matrix = None
        self.is_compiled = False
        self.output_dim = output_dim
        
        # Expose for tool compatibility (VocabCacheGenerator etc)
        self.model = getattr(self.embedder, "model", None)
        self.processor = getattr(self.embedder, "processor", None)

    def load_vocab_cache(self, json_path: str):
        if not os.path.exists(json_path):
            self.log_callback(f"No vocab cache found at {json_path}")
            return
        
        self.log_callback(f"Loading vocabulary mapping from {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        tensor_path = None
        words = None

        if isinstance(metadata, dict):
            words = metadata.get("words")
            tensor_path = metadata.get("tensor_path")
        elif isinstance(metadata, list):
            words = metadata

        if not isinstance(words, list) or not words:
            raise ValueError(f"Invalid vocab cache JSON format: expected 'words' list in {json_path}")

        self.vocab_words = [str(w) for w in words]

        # Determine tensor file path (supports both legacy `<same>.pt` and explicit `tensor_path`).
        if tensor_path:
            pt_path = os.path.join(os.path.dirname(json_path), str(tensor_path))
        else:
            pt_path = json_path.replace(".json", ".pt")

        if not os.path.exists(pt_path):
            self.log_callback(f"Missing tensor cache file: {pt_path}")
            return

        self.log_callback(f"Loading tensor cache: {pt_path}")
        # Load on CPU first to avoid peak GPU memory spikes during dtype conversion.
        matrix = torch.load(pt_path, map_location="cpu", weights_only=True)
        if not isinstance(matrix, torch.Tensor):
            raise TypeError(f"Tensor cache is not a torch.Tensor: {pt_path}")

        # Matryoshka-style truncation to reduce VRAM/compute. Re-normalize after truncation.
        if self.output_dim is not None and matrix.ndim == 2:
            if matrix.shape[1] > self.output_dim:
                matrix = matrix[:, : self.output_dim].contiguous()
                matrix = F.normalize(matrix.float(), p=2, dim=-1)
            elif matrix.shape[1] < self.output_dim:
                self.log_callback(
                    f"Warning: vocab matrix dim={matrix.shape[1]} < requested output_dim={self.output_dim}; keeping full dim."
                )

        # Enforce normalization for all cases. 
        # Similarity matching (Dot product) requires normalized vectors to behave like Cosine Similarity.
        matrix = F.normalize(matrix.float(), p=2, dim=-1)
        
        if matrix.dtype != self.dtype:
            matrix = matrix.to(self.dtype)

        self.vocab_matrix = matrix.to(self.device, non_blocking=(self.device.type == "cuda"))

    def prepare_vocabulary(self, fallback_vocab: List[str]):
        self.log_callback(f"Preparing dynamic vocabulary ({len(fallback_vocab)} tags)...")
        self.vocab_words = fallback_vocab
        inputs = [{"text": w} for w in fallback_vocab]
        self.vocab_matrix = self.embedder.process(inputs, normalize=True)
        self.log_callback("Vocabulary precomputation complete.")

    def predict_batch(
        self,
        images: List[Image.Image],
        threshold: float = 0.3,
        max_tags: int = 50,
        **kwargs
    ) -> List[List[Tuple[str, float]]]:
        if self.vocab_matrix is None:
            return [[] for _ in images]

        # self.log_callback(f"[Qwen Engine] Entering Embedding block for {len(images)} images...")
        
        inputs = [{"image": img} for img in images]
        img_embeddings = self.embedder.process(inputs, normalize=True)
        
        # Matryoshka-style truncation for image embeddings to match truncated vocab matrix
        if self.output_dim is not None and img_embeddings.shape[1] > self.output_dim:
            img_embeddings = img_embeddings[:, :self.output_dim].contiguous()
            img_embeddings = F.normalize(img_embeddings, p=2, dim=-1)

        # Sync dtypes for dot product
        if img_embeddings.dtype != self.vocab_matrix.dtype:
            img_embeddings = img_embeddings.to(self.vocab_matrix.dtype)

        # Similarity matching (Dot product on normalized vectors)
        scores = img_embeddings @ self.vocab_matrix.T
        
        results = []
        for i in range(len(images)):
            image_scores = scores[i]
            # Filter and sort
            top_indices = torch.where(image_scores > threshold)[0]
            top_scores = image_scores[top_indices]
            
            # Sort by score DESC
            sorted_indices = top_indices[torch.argsort(top_scores, descending=True)]
            
            # Sort by score DESC
            sorted_indices = top_indices[torch.argsort(top_scores, descending=True)]
            
            tags = []
            for idx in sorted_indices[:max_tags]:
                tags.append((self.vocab_words[idx], float(scores[i, idx])))
            results.append(tags)
            
        return results
