import json
import os
from PIL import Image
from processing import SGLangModel

def test_generative():
    # Load config from the root directory
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Target image from EXTRASFW - Copy
    img_path = r"C:\Users\chatr\Pictures\EXTRASFW - Copy\3657168.jpg"
    if not os.path.exists(img_path):
        # Fallback to local sample if needed
        print(f"Warning: {img_path} not found. Checking local root...")
        img_path = "3657168.jpg"
        
    if not os.path.exists(img_path):
        print("Error: Target image not found.")
        return
        
    print(f"Testing Stabilized SGLang Pro Logic on {os.path.basename(img_path)}...")

    # Initialize and Load
    model = SGLangModel("sglang", "", config)
    model.load()
    
    img = Image.open(img_path)
    
    # model.infer handles the async burst + session pooling internally
    results = model.infer([img])
    
    if results:
        print("\n--- FINAL PARSED OUTPUT ---")
        print(json.dumps(results[0], indent=4))
        
        # Check for error in result
        if "error" in results[0]:
            print(f"\n[!] Failure Detected: {results[0]['error']}")
        else:
            print("\n[SUCCESS] Vision restored and throughput optimized!")
    else:
        print("Error: No results.")
    
    model.unload()

if __name__ == "__main__":
    test_generative()
