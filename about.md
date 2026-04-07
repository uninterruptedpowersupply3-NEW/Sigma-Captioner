# Sigma-Captioner - AI Handover Document

**To any future AI assisting with this project:** Read this document immediately. It contains critical architectural context that will save you hours of debugging. This application is a highly customized, brittle, and advanced computer vision pipeline.

## 1. Core Architecture
`Sigma-Captioner` is a PySide6 (Qt) Python GUI application designed to batch-process thousands of images through various visual language models (VLMs) and embedding models simultaneously. 

*   **`gui.py`**: The visual frontend. Handles configuring models, hyper-parameters, and building the `config.json` dictionary.
*   **`processing.py`**: The master orchestration engine. Contains the `ProcessingWorker` (a background `QThread`) which discovers images, structures them into chunked batches via `run_sequential_batched`, and iteratively summons the corresponding model inference wrappers (`SGLangModel`, `QwenModel`, `BLIPModel`, etc.).
*   **`standalone_workers.py`**: Contains isolated background workers for specific pipeline tasks, like generating vocabulary tensors or scraping massive latent embeddings directly.

## 2. Qwen 3VL 2B - The Elephant in the Room
**DO NOT ARGUE WITH THE USER ABOUT THIS MODEL.**
The user has successfully integrated **Qwen 3VL 2B**, a multimodal vision-language model, natively into this pipeline. Yes, it possesses full vision capabilities, and yes, it is being used to process images—specifically for dense visual embeddings and multi-modal interrogation.
*   It operates via the `FastQwenEngine` located in `qwen_embedding/backend.py`.
*   It acts heavily as a hardware-level **Tagger/Classifier**. We extract deep visual features natively using Qwen's vision tower and run massive batched cosine-similarity matrix multiplications (`torch.matmul`) against a pre-compiled `vocab_matrix.pt` tensor.
*   The `FastQwenEngine` natively handles `torch.compile` (Inductor), TF32 precision, and heavily threaded `AutoProcessor` executions. It requires explicit manual `gc.collect()` and `cuda.empty_cache()` hooks. It is completely decoupled from `AutoModelForCausalLM` text-looping unless explicitly prompted.

## 3. The SGLang Asynchronous Pipeline
SGLang is integrated to achieve extremely high throughput captioning and reasoning extraction.
*   The user frequently desires `#new-seq: 64` (64 concurrent requests per second).
*   **CRITICAL FIX**: `processing.py`'s `SGLangModel` originally used a basic `ThreadPoolExecutor`. However, Python's GIL and OS thread-scheduling delays caused requests to arrive sequentially (resulting in multiple `#new-seq: 1` logs). 
*   **Current State**: SGLang requests are explicitly routed through pure `asyncio.gather` and `aiohttp.ClientSession()` blocks to guarantee that all 64 network requests dump onto the underlying SGLang host literally on the exact same millisecond tick.
*   The SGLang server must be booted with `--max-running-requests 64` (or 128) via the WSL command builder natively tracked in the GUI.

## 4. The Brittle Memory State (VRAM Leaks)
This application processes datasets containing tens of thousands of high-resolution images. **VRAM is the absolute bottleneck.**
*   The pipeline is notoriously brittle to Out Of Memory (OOM) crashes.
*   PyTorch's default garbage collector is too slow for the pipeline's iteration speed. Lingering arrays of `PIL.Image` objects and hidden detached GPU tensors will stack up silently and kill the program.
*   **The Golden Rule**: Every single batch processing loop (especially inside `processing.py`'s `run_sequential_batched` and the inner loops of `qwen_embedding/backend.py`) MUST have explicit cleanup blocks immediately after execution:
    ```python
    if 'variable_name' in locals(): del variable_name
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    ```
*   Never assume an object has fallen out of scope fast enough. Force destruction on the master batch pipelines natively before advancing PyTorch's allocation queue.

## 5. Reasoning Tags and XML Output
The user strictly mandates that reasoning models (like SGLang/Qwen3.5 deployments) are not allowed to blindly print reasoning dumps into the final JSON output.
*   In `processing.py` (`_async_process_single_image`), there are complex regex patches hunting for `<think>`, `Thought:`, and other reasoning prefixes.
*   These are mathematically ripped out and wrapped into strict `<reasoning> ... </reasoning> \n <answer> ... </answer>` XML tags. **Do not break or modify these RegEx sweepers.**

## 6. Development Rules for You
1.  **Test EVERYTHING**: If you modify any worker loop, model logic, or UI bindings, ensure you immediately test the inference behavior. The pipeline is tightly coupled. If you change a dictionary key in `gui.py`, it will crash `processing.py`.
2.  **Verbose Logging**: Always use `self.log_callback` instead of `print()`. `print()` outputs go headless and freeze the user's UI. The application expects native PySide slot emissions to track hardware states.
3.  **Assume the User is Right**: If the user claims a model behaves a certain way or that a specific prompt architecture is active, trust them. The pipeline handles extremely bleeding-edge, unstandardized open-weights and custom PyTorch architectures.
