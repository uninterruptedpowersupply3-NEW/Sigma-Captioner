import os
import time
import json
from processing import ProcessingWorker

def test_firehose_run():
    # Setup mock config
    image_dir = os.path.join(os.getcwd(), "test_images_mock")
    os.makedirs(image_dir, exist_ok=True)
    
    # Create 5 mock images
    from PIL import Image
    for i in range(5):
        img_path = os.path.join(image_dir, f"mock_{i}.jpg")
        if not os.path.exists(img_path):
            Image.new('RGB', (100, 100), color='red').save(img_path)

    config = {
        "image_dir": image_dir,
        "models_enabled": {"sglang": True},
        "sglang_url": "http://127.0.0.1:30000/v1/chat/completions",
        "sglang_concurrency": 2, # Small for testing
        "sglang_system_context": "You are a helpful assistant.",
        "resume_processing": False, 
        "model_dir": os.getcwd()
    }

    print("--- Starting Firehose Mock Test ---")
    worker = ProcessingWorker(config)
    
    # Mock log signal to print
    worker.log.connect(lambda msg: print(f"[WORKER LOG] {msg}"))
    
    # Override model loading to avoid real weights
    from processing import SGLangModel
    class MockSGLangModel(SGLangModel):
        def load(self): print("Mock SGLang Loaded")
        def unload(self): print("Mock SGLang Unloaded")
        def infer(self, images, **kwargs):
            print(f"Mock Inferring on {len(images)} images")
            return [{"qa_pairs": [{"question": "Q", "answer": "A"}]}] * len(images)

    worker.model_map["SGLang"] = MockSGLangModel

    worker.run()
    
    # Cleanup
    import shutil
    # shutil.rmtree(image_dir) # Keep for debugging if needed

if __name__ == "__main__":
    test_firehose_run()
