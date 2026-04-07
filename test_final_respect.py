import asyncio
import json
import os
from PIL import Image
from processing import SGLangModel

async def test_final():
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Force a sample image
    img_path = r"C:\Users\chatr\Pictures\EXTRASFW - Copy\3657168.jpg"
    print(f"Testing SGLang Native /generate on {os.path.basename(img_path)}...")
    print(f"Using GUI max_tokens: {config.get('sglang_max_tokens')}")

    model = SGLangModel("sglang", "", config)
    model.load()
    
    img = Image.open(img_path)
    results = model.infer([img])
    
    if results and len(results) > 0:
        res = results[0]
        if "error" in res:
            print(f"\nFAILURE: {res['error']}")
        else:
            print("\nSUCCESS: Caption Generated!")
            print(json.dumps(res, indent=4))
    else:
        print("\nERROR: No results.")

if __name__ == "__main__":
    asyncio.run(test_final())
