# 📸 Sigma-Captioner

A somewhat optimized implementation of lightweight and popular vision-language models.

This application is designed to generate image-text pairs to make sorting images easier, or it can be used for bulk-captioning large datasets.

### ⚙️ How It Works

  * **Smart Parsing:** The outputs of models like SmolVLM will only be saved/printed if they follow the expected format. Otherwise, the compute is discarded to prevent dataset contamination. *(Note: This strict parsing can be changed manually in the settings).*
  * **Easy Interface:** The **Download** button fetches all selected models locally, and the **Start** button immediately begins bulk inference on your dataset.

<img width="2113" height="264" alt="image" src="https://github.com/user-attachments/assets/f8e058e5-ffc2-46e4-9b4e-73f6b1a42ab2" />

<img width="1786" height="1345" alt="image" src="https://github.com/user-attachments/assets/2971e951-c218-4a33-a840-93abddc81024" />

<img width="1000" height="520" alt="image" src="https://github.com/user-attachments/assets/1f51ea80-6da3-47ca-81e0-afafa1f8104a" />

-----

## Inspired by CLIP-Interrogator

The Qwen3-VL-Embedding system is heavily inspired by the classic CLIP-Interrogator workflow. It extracts high-dimensional visual features from your images and mathematically compares them against a precalculated Vocabulary Matrix—a massive cache of text-based embeddings—by performing similarity matching via a dot product on normalized vectors, which mathematically behaves perfectly like Cosine Similarity.

This implementation is engineered for extreme speed, easily achieving 100+ captions per second on an RTX 3070 Ti. To feed the GPU this fast, the pipeline utilizes highly optimized Windows GPU-based image processing. It performs Dynamic Resolution Bucketing—rapidly scanning image headers to extract dimensions without fully decoding the files. This allows the system to instantly group batches by uniform resolutions and snap them precisely to a 28-pixel grid, maximizing the Qwen vision encoder's throughput.

📦 Out-of-the-Box Setup for Qwen
For the Qwen embedding system to work immediately out of the box, you will need a precalculated vocabulary matrix.
👉 Download the Official Precalculated Matrices (Hugging Face)
https://huggingface.co/UPShf/Vocabulary-Qwen3-VL-Embedding-2B

Making your own: If you prefer to use your own custom tags or dictionary, you can generate a matrix manually using the built-in Preprocessing Workstation tab. Thanks to the optimized batching engine, compiling a massive custom dictionary from raw text takes less than 20 minutes on standard hardware.

🙏 Acknowledgements & Datasets
The official precalculated vocabulary matrices were generated using the following incredible open-source datasets. Huge thanks to their creators:

English [dwyl/english-words](https://github.com/dwyl/english-words)

Anime/Visual [cagliostrolab/860k-ordered-tags-json](https://huggingface.co/datasets/cagliostrolab/860k-ordered-tags-json) (Danbooru tags)

-----

## 🛠️ Installation & Setup (Windows)

We have provided automated scripts to make setup painless.

1.  **Run the Setup Script:** Double-click `setup.bat`. This will automatically create your virtual environment and install all necessary dependencies (including `qwen-vl-utils`).
2.  **Optional Performance Boost (Florence):** For faster generation, it is highly recommended to install Flash Attention. Download the pre-compiled wheel for your system and install it manually from here https://github.com/wildminder/AI-windows-whl
:
    ```bash
    pip install <location of flash attn wheel>
    ```
    *Example:* `flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp310-cp310-win_amd64.whl`

3.  **Launch the App:** Double-click `run.bat` (or run `python main.py` with your venv activated) to start the UI.

-----

## 🚀 SGLang Integration via WSL

To use the high-speed SGLang backend, you must run the server from a Linux environment using WSL (Windows Subsystem for Linux).

**Step 1:** Open your WSL terminal and install SGLang inside your WSL Python environment.
**Step 2:** Launch the SGLang server.

> [\!WARNING]
> **EXAMPLE COMMAND ONLY:** Do not blindly copy and paste the command below\! You MUST change the `--model-path` to point to wherever your model is actually located on your mounted Windows drive.

```bash
# Example WSL launch command
WSL && cd && cd v* && cd bin && source activate && python3 -m sglang.launch_server --model-path "/mnt/c/Users/UPS/Documents/Tech/text-generation-webui/user_data/models/Qwen3.5-2B" --context-length 1024  --mem-fraction-static 1.01  --max-running-requests 64  --enable-torch-compile --kv-cache-dtype bf16 --attention-backend flashinfer --port 30000 --schedule-policy fcfs --tp-size 1
```

-----
**Legacy Support:** For seamless integration with older dataset structures, both of these models support a **Legacy Support** toggle in the settings. Enabling this rewrites their JSON properties to perfectly mimic the structures of `wd_tagger` and `smolvlm` outputs respectively, ensuring zero downtime with your existing sorting tools.

-----

## 📊 Supported Models

| Model Name | Status |
| :--- | :--- |
| `Salesforce/blip-vqa-base`                     | ✅ Works |
| `Salesforce/blip-image-captioning-large`       | ✅ Works |
| `microsoft/Florence-2-large-ft`                | ✅ Works |
| `microsoft/git-large-textvqa`                  | ✅ Works |
| `HuggingFaceTB/SmolVLM-256M-Instruct`          | ✅ Works |
| `SmilingWolf/wd-eva02-large-tagger-v3`         | ✅ Works |
| `Qwen/Qwen3-VL-Embedding-2B`                   | ✅ Works |
| `LM-Sys/SGLang` (Local Endpoint)               | ✅ Works |
| `SGLang` (WSL - ANY MODEL)                     | ✅ Works |
| `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`        | ⚠️ In Development |
| `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`        | ⚠️ In Development |
| `vikhyatk/moondream2`                          | ⚠️ In Development |
| `fancyfeast/llama-joycaption-beta-one-hf-llava`| ⚠️ In Development |
