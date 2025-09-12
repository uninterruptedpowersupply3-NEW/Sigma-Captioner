from huggingface_hub import hf_hub_download

class SmolVLMWrapper:
    def __init__(self, model_path_or_id, config, repo_type="model"):
        """
        model_path_or_id: local folder path or Hugging Face model ID
        config: dict with model settings
        repo_type: 'model' or 'dataset'
        """
        self.config = config

        # Check if path exists and has safetensors
        if not os.path.exists(model_path_or_id) or not any(f.endswith(".safetensors") for f in os.listdir(model_path_or_id)):
            print(f"[CLI] Missing model files. Downloading '{model_path_or_id}' in safetensors format...")
            # Download the main .safetensors file
            file_path = hf_hub_download(repo_id=model_path_or_id, filename="pytorch_model.safetensors")
            # Create folder if missing
            os.makedirs(model_path_or_id, exist_ok=True)
            # Move downloaded file to target folder
            os.rename(file_path, os.path.join(model_path_or_id, "pytorch_model.safetensors"))
            print(f"[CLI] Download complete. Saved to {model_path_or_id}")

        self.model_path = model_path_or_id
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=DTYPE,
            trust_remote_code=True
        ).to(DEVICE).eval()
        try:
            self.model = torch.compile(self.model)
            self.optim = "torch.compile"
        except Exception:
            self.optim = "none"
        self.attn_impl = "default"
        self.precision = str(DTYPE)
        self.use_cuda_graph = False
        self.is_cuda_graph_warmed_up = False
        self.static_inputs = None
        self.graph = None
        print(f"[CLI] SmolVLM loaded. Precision: {self.precision}, Attention: {self.attn_impl}, Optimizer: {self.optim}")
