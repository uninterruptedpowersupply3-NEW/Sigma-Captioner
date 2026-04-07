# Fast Qwen3-VL Embedding Tagger 🚀

An optimized, dual-stage visual tagging pipeline built on top of [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B). By splitting the traditional Vision-Language Model process into two distinct stages (feature extraction and embedding matching), this software achieves massive throughput capable of evaluating millions of image latents against tens of thousands of tags **extremely fast**.

## 🌟 Key Features

* **Massive Vocabulary Matrix:** Dynamically construct an immense tagging dictionary (e.g., 6,000+ Danbooru tags + 370k DWYL English words) entirely inside system RAM/VRAM as a precomputed dense vector matrix.
* **Dual-Stage Architecture:** 
  * **Stage 1 (Cache Maker):** Encode bulk image folders into dense contextual `.pt` tensors using `torch.compile` and Inductor.
  * **Stage 2 (Inference):** Completely bypass the VLM/ViT bottleneck. Feed raw image `.pt` latents directly into a fast CPU/GPU cosine-similarity tensor matrix to evaluate tags thousands of times faster than standard inference.
* **Smart Dynamic Grouping:** Automatically peeks at image file headers to group variable resolutions into uniform bounding boxes (quantized to 28-pixel grid patches) to maximize PyTorch Inductor kernel efficiency and prevent recompilations.
* **Multi-threaded Preprocessing Pool:** Utilizes non-blocking Python `concurrent.futures` to overlap CPU I/O resizing heavily with active GPU execution.

## ⚙️ Installation

You need Python 3.10+ and a CUDA-capable GPU.

1. Clone the repository.
2. Install the necessary PyTorch and HuggingFace libraries:
   ```bash
   pip install uv
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   uv pip install transformers accelerate pillow numpy imagesize
   # Important: You must install the Qwen VL utils for optimal vision processing
   uv pip install qwen-vl-utils
   ```

## 🖥️ Workflows

The toolkit provides two Graphical User Interfaces (GUIs) tailored for the pipeline stages:

### `run_cache_maker.bat` (The Preprocessor Station)
Use this UI for the heavy lifting:
1. **📚 Build Vocab Cache:** Point it to your `data.json` to extract tags, filter by occurrences, and export a gigantic dense tensor matrix `.pt` and `.json` cache.
2. **🖼️ Extract Image Latents:** Select a massive folder of raw images (`.jpg`, `.png`). The Engine will process them through Qwen3-VL's compiled Vision encoder, generate standalone `.pt` feature tensors for every image, and save them.

### `run.bat` (The Inference Engine)
Use this UI for the ultimate tagging speed:
1. **Load Cache File:** Initialize the blazing-fast vocabulary matrix cache into memory.
2. **Select Data Folder:** Point it at the folder containing your generated `.pt` embedding latents.
3. **Execute:** The engine detects the `.pt` tensors, completely bypasses the Vision transformer model, and performs instantaneous Cosine Similarity matrix multiplication. Output tags are saved as `.txt` files directly alongside your images.

*(Note: The Inference Engine can also accept standard image formats, but running `.pt` embeddings offers significantly higher FPS since the ViT logic is skipped.)*

## 🏗️ Architecture Under the Hood

Unlike generative VLMs which decode visual tokens step-by-step into text, `Qwen3-VL-Embedding` outputs dense vectors that share the exact mathematical latent space as pure text embeddings.

**Instead of `Image -> VLM -> Text`**, we optimize it to:
1. `Image -> ViT -> Image Vector (.pt)`
2. `Text -> Model -> Text Vector Matrix (.pt)`
3. `Image Vector @ Text Vector Matrix.t() = Similarities`

The UI automatically intercepts the `torch.cuda.empty_cache()` dangling graph bugs, safely patches Inductor via `torch._dynamo.config.suppress_errors = True` to navigate unbacked SymInt geometry limitations on Windows, and implements highly parallelized data-shuttling to saturate the GPU.

## 🤝 Acknowledgments

* The [Qwen](https://huggingface.co/Qwen) Team for producing `Qwen3-VL-Embedding-2B`.
* Built utilizing standard PyTorch `torchaudio`, `torchvision`, and standard `Transformers` libraries.
