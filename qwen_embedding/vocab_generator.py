import torch
import json
import os
import urllib.request
from backend import FastQwenEngine

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from concurrent.futures import ThreadPoolExecutor

class VocabCacheGenerator:
    """
    Handles generation of massive vocabulary caches (10k to 300k+ words).
    Dumps to a fast-loading PyTorch tensor cache and a JSON word map.
    """
    def __init__(self, precision="bf16", device="cuda"):
        # We spawn the FastQwenEngine exclusively to encode text.
        self.engine = FastQwenEngine(precision=precision, device=device)
        self.device = device
        
    def download_words(self, url: str, target_file: str):
        """Helper to download raw text lists."""
        print(f"Downloading wordlist from {url}...")
        urllib.request.urlretrieve(url, target_file)
        print("Download complete.")
        
    def generate_and_save(self, words_source: list, out_pt_path: str, out_json_path: str, batch_size=2048):
        """
        Batches the word encoding so the GPU doesn't OOM on 100k+ strings.
        Saves a compiled .pt matrix and a structured .json reference map.
        """
        print(f"Starting massive vocab encoding for {len(words_source)} tags...")
        
        all_embeddings = []
        words_source = [str(w).strip() for w in words_source if str(w).strip()]
        
        # Freezing the backend model for pure inference
        self.engine.model.eval()
        
        with torch.inference_mode():
            for i in range(0, len(words_source), batch_size):
                batch = words_source[i:i+batch_size]
                print(f"Encoding batch {i}/{len(words_source)}...")
                
                # Standard conversion
                inputs = self.engine.processor(text=batch, return_tensors="pt", padding=True).to(self.device)
                outputs = self.engine.model(**inputs)
                
                if hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state[:, -1, :]
                else:
                    embeddings = outputs
                    
                # Store entirely on CPU RAM until saving to prevent GPU OOM
                embeddings = torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)
                all_embeddings.append(embeddings.cpu())
                
                # Extreme VRAM Cleanup to prevent creeping memory fragmentation
                del inputs
                del outputs
                del embeddings
                torch.cuda.empty_cache()
                
        # Concat fully
        final_matrix = torch.cat(all_embeddings, dim=0)
        
        # Save Tensor Matrix
        torch.save(final_matrix, out_pt_path)
        print(f"Vocabulary matrix saved to {out_pt_path} with shape {final_matrix.shape}")
        
        # Save JSON Dictionary mapping index to word
        json_data = {
            "count": len(words_source),
            "words": words_source,
            "tensor_path": os.path.basename(out_pt_path)
        }
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)
            
        print(f"JSON Metadata saved to {out_json_path}")
        print("Vocab Cache Generation Successful.")
        
    def build_hybrid_danbooru_english_vocab(self, danbooru_json_path: str, out_pt_path: str, out_json_path: str, min_occurrences=200, use_english_dict=True, batch_size=64):
        """
        Parses exactly the format of data.json. Extracts 'character', 'copyright', 'artist', 'general'.
        Counts tags, filters out any appearing less than `min_occurrences` times.
        Downloads the 370k DWYL English list (if enabled), merges them perfectly, removes duplicates, and generates the cache!
        """
        from collections import Counter
        import tempfile
        
        tag_counts = Counter()
        
        print(f"Parsing Danbooru dataset: {danbooru_json_path}")
        if os.path.exists(danbooru_json_path):
            with open(danbooru_json_path, 'r', encoding='utf-8') as f:
                import json
                try:
                    data = json.load(f)
                    for item in data:
                        # Extract tags from relevant fields
                        for field in ['character', 'copyright', 'artist', 'general']:
                            if field in item and item[field]:
                                # Split by comma
                                tags = [t.strip() for t in item[field].split(", ")]
                                tag_counts.update(tags)
                except Exception as e:
                    print(f"Failed to parse JSON directly: {e}. Try splitting the file if it's multiple GBs.")
        else:
            print(f"Warning: {danbooru_json_path} not found. Skipping Danbooru extraction.")
            
        # Filter Danbooru Tags > min_occurrences
        filtered_danbooru = [tag for tag, count in tag_counts.items() if count >= min_occurrences]
        print(f"Found {len(tag_counts)} unique visual tags. Kept {len(filtered_danbooru)} tags appearing >= {min_occurrences} times.")
        
        # Download DWYL English Words
        if use_english_dict:
            tmp_eng_path = os.path.join(tempfile.gettempdir(), "words_alpha.txt")
            if not os.path.exists(tmp_eng_path):
                self.download_words("https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt", tmp_eng_path)
                
            with open(tmp_eng_path, 'r', encoding='utf-8') as f:
                english_words = [line.strip() for line in f if line.strip()]
                
            print(f"Loaded {len(english_words)} standard English dictionary words.")
        else:
            english_words = []
            print("Skipped DWYL English dictionary merge.")
        
        # Merge and remove duplicates (preserving pure string matching)
        hybrid_set = set(filtered_danbooru)
        hybrid_set.update(english_words)
        
        final_hybrid_vocab = list(hybrid_set)
        # Optional sorting to group 1girl, 1boy, character names, series, etc loosely, but sets are inherently random.
        final_hybrid_vocab.sort()
        
        print(f"--- Hybrid Fusion Complete ---")
        print(f"Total Unique Tokens Extracted: {len(final_hybrid_vocab)}")
        
        # Save to .txt explicitly for the user
        txt_path = out_json_path.replace(".json", "_words.txt")
        print(f"Saving explicitly to text list: {txt_path}...")
        with open(txt_path, 'w', encoding='utf-8') as f:
            for word in final_hybrid_vocab:
                f.write(f"{word}\n")
                
        print(f"Proceeding to massive batched GPU embedding cache generation...")
        
        # Automatically trigger the heavy batched GPU engine computation
        self.generate_and_save(
            words_source=final_hybrid_vocab,
            out_pt_path=out_pt_path,
            out_json_path=out_json_path,
            batch_size=batch_size
        )

if __name__ == "__main__":
    generator = VocabCacheGenerator(precision="int8")
    
    # Run the hybrid fusion dynamically
    generator.build_hybrid_danbooru_english_vocab(
        danbooru_json_path="data.json", # The file you mentioned
        out_pt_path="vocab_hybrid_matrix.pt",
        out_json_path="vocab_hybrid_meta.json",
        min_occurrences=200,
        use_english_dict=True,
        batch_size=64
    )
