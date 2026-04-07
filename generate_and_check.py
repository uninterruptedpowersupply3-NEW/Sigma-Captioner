import asyncio
import json
import os
from PIL import Image
from processing import SGLangModel

async def test_and_save():
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    img_path = r"C:\Users\chatr\Pictures\EXTRASFW - Copy\3657168.jpg"
    json_path = r"C:\Users\chatr\Pictures\EXTRASFW - Copy\3657168.json"
    
    model = SGLangModel("sglang", "", config)
    model.load()
    
    img = Image.open(img_path)
    print(f"Captioning {os.path.basename(img_path)}...")
    results = model.infer([img])
    
    if results:
        res = results[0]
        # Simulate the sidecar save logic
        sidecar = {
            "image_path": img_path,
            "sglang": res
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=4)
        print(f"Saved results to {json_path}")
        print("\n--- CONTENT PREVIEW ---")
        print(json.dumps(res, indent=4))
    else:
        print("Error: No results.")

if __name__ == "__main__":
    asyncio.run(test_and_save())
