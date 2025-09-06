import os
import json
import time
import traceback
from PIL import Image
# --- ADD THIS IMPORT ---
from PyQt6.QtCore import QThread, pyqtSignal
from torch.utils.data import Dataset, DataLoader
# --- END OF ADDITION ---
import torch
import gc
from collections import defaultdict

# ... (imports for models are fine)
from models import (
    BlipModel, FlorenceModel, ClipInterrogatorModel,
    JoyCaptionModel, GitModel
)
from wd_tagger_app2 import WDTaggerApp2Model
from moondream_model import MoondreamModel
from smolvlm_model import SmolVLMModel
from utils import get_model_path

Image.MAX_IMAGE_PIXELS = None

# --- START OF MODIFICATION 1: ADD DATASET AND COLLATE FUNCTION ---

class ImageCaptioningDataset(Dataset):
    """
    A PyTorch Dataset to handle loading images from a list of paths.
    This is what the DataLoader's worker processes will use.
    """
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            # Open the image using PIL
            with Image.open(path) as img:
                # We return a copy of the image object and its original path
                return img.copy(), path
        except Exception:
            # If an image is corrupt or cannot be opened, return None.
            # Our collate_fn will handle filtering these out.
            return None, None

def collate_fn(batch):
    """
    A custom collate function to filter out failed image loads (None values)
    before they are passed to the model.
    """
    # Filter out samples where the image failed to load
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        # If the whole batch failed, return empty lists
        return [], []
    
    # "Unzip" the batch of (image, path) tuples into two separate lists
    images, paths = zip(*batch)
    return list(images), list(paths)

# --- END OF MODIFICATION 1 ---


class ProcessingWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(str)
    log = pyqtSignal(str)
    
    def __init__(self, config):
        super().__init__()
        # ... (the rest of __init__ is unchanged)
        self.config = config
        self._is_running = True
        
        self.model_map = {
            "CLIP_Interrogator": ClipInterrogatorModel, "BLIP": BlipModel, 
            "Florence-2": FlorenceModel, "JoyCaption": JoyCaptionModel, 
            "GIT": GitModel, "WD_Tagger": WDTaggerApp2Model,
            "Moondream": MoondreamModel, "SmolVLM": SmolVLMModel,
        }
        self.json_key_map = {
            "CLIP_Interrogator": "clip_interrogator", "BLIP": "blip", 
            "Florence-2": "florence", "JoyCaption": "llava", "GIT": "git", 
            "WD_Tagger": "wd_tagger", "Moondream": "moondream", 
            "SmolVLM": "smolvlm",
        }
        
        self.general_questions = []
        self.moondream_questions = []
        self.image_question_map = {}
        self.moondream_image_question_map = {}

    # ... (stop, run, and question loading methods are unchanged) ...
    def stop(self):
        self._is_running = False
        self.log.emit("Processing stop requested. Finishing current batch...")

    def run(self):
        start_time = time.time()
        self.log.emit("Discovering images...")

        if self.config.get('use_question_file'):
            try:
                with open(self.config['question_json_path'], 'r', encoding='utf-8') as f:
                    self.general_questions = json.load(f)
                if not isinstance(self.general_questions, list) or not self.general_questions:
                    self.log.emit("Warning: General question file is invalid or empty. Using common question.")
                    self.general_questions = [self.config.get('common_question')]
                else:
                    self.log.emit(f"Loaded {len(self.general_questions)} general questions.")
            except Exception as e:
                self.log.emit(f"Error loading general question file: {e}. Using common question.")
                self.general_questions = [self.config.get('common_question')]
        else:
            self.general_questions = [self.config.get('common_question')]

        if self.config.get('use_moondream_question_file'):
            try:
                with open(self.config['moondream_question_json_path'], 'r', encoding='utf-8') as f:
                    self.moondream_questions = json.load(f)
                if not isinstance(self.moondream_questions, list) or not self.moondream_questions:
                    self.log.emit("Warning: Moondream question file is invalid or empty. Falling back to general questions.")
                    self.moondream_questions = self.general_questions
                else:
                    self.log.emit(f"Loaded {len(self.moondream_questions)} Moondream-specific questions.")
            except Exception as e:
                self.log.emit(f"Error loading Moondream question file: {e}. Moondream will use general questions.")
                self.moondream_questions = self.general_questions
        else:
            self.moondream_questions = self.general_questions

        image_paths = self._discover_images()
        images_to_process = self._filter_images(image_paths)
        if not images_to_process:
            self.log.emit("No new images to process."); self.finished.emit("Finished."); return

        for i, path in enumerate(images_to_process):
            self.image_question_map[path] = self.general_questions[i % len(self.general_questions)]
            self.moondream_image_question_map[path] = self.moondream_questions[i % len(self.moondream_questions)]

        final_results = self.run_sequential_batched(images_to_process)
        self.log.emit("--- All models processed. Saving JSON files... ---")
        for i, path in enumerate(final_results.keys()):
            if not self._is_running: break
            self.progress.emit(i + 1, len(final_results), f"Saving {os.path.basename(path)}")
            question_used = self.moondream_image_question_map.get(path) if 'moondream' in final_results[path] and self.config.get('use_moondream_question_file') else self.image_question_map.get(path)
            self._save_final_json(path, final_results[path], question_used)
        
        duration = time.time() - start_time
        self.finished.emit(f"Processing finished in {duration:.2f} seconds.")


    # --- START OF MODIFICATION 2: REPLACE THE run_sequential_batched METHOD ---
    def run_sequential_batched(self, images_to_process):
        self.log.emit("--- Starting sequential batch processing with DataLoader ---")
        self.models = {}
        final_results = defaultdict(dict)
        total_images = len(images_to_process)
        
        try:
            for model_key, model_class in self.model_map.items():
                json_key = self.json_key_map[model_key]
                if not self.config['models_enabled'].get(json_key): continue
                if not self._is_running: break
                
                model_path = get_model_path(model_key, self.config)
                self.log.emit(f"[{model_key}] Loading from {model_path}...")
                
                model = model_class(model_key, model_path, self.config)
                model.set_log_callback(self.log.emit)
                model.load()
                self.models[model_key] = model
                
                batch_size = self.config['model_specific_batch_sizes'].get(json_key, 1)
                self.log.emit(f"[{model_key}] Starting processing with batch size {batch_size} and parallel workers...")

                # Create the Dataset and DataLoader
                dataset = ImageCaptioningDataset(images_to_process)
                data_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    prefetch_factor=5,
                    num_workers=8,  # Use multiple CPU cores to load data in the background
                    pin_memory=True, # Speeds up CPU to GPU data transfer
                    collate_fn=collate_fn # Our function to handle corrupt images
                )

                total_processed_for_model = 0
                # The new, efficient processing loop
                for batch_images, batch_paths in data_loader:
                    if not self._is_running: break
                    if not batch_images: continue # Skip if the entire batch was corrupt

                    total_processed_for_model += len(batch_paths)

                    # Get the questions for the current valid batch
                    if model_key == "Moondream":
                        questions_for_batch = [self.moondream_image_question_map.get(path) for path in batch_paths]
                    else:
                        questions_for_batch = [self.image_question_map.get(path) for path in batch_paths]
                    
                    try:
                        self.progress.emit(total_processed_for_model, total_images, f"Processing {len(batch_images)} images with {model_key}...")
                        batch_outputs = self._run_inference_for_model(model, model_key, json_key, batch_images, questions_for_batch)
                    except Exception as e:
                        self.log.emit(f"---!!! FATAL BATCH ERROR with {model_key} !!!---")
                        self.log.emit(f"ERROR: {e}\n{traceback.format_exc()}")
                        self.log.emit("---!!! SKIPPING THIS BATCH AND CONTINUING !!!---")
                        batch_outputs = [{"Error": f"Fatal batch error: {e}"}] * len(batch_images)

                    for idx, output in enumerate(batch_outputs):
                        # The batch_paths list is now guaranteed to be the correct size
                        final_results[batch_paths[idx]].update({json_key: output})

                self.models[model_key].unload()
                del self.models[model_key]

        finally:
            self._cleanup()
        return final_results
    # --- END OF MODIFICATION 2 ---

    # ... (the rest of the file, from _run_inference_for_model onwards, is unchanged) ...
    def _run_inference_for_model(self, instance, model_key, json_key, batch_images, questions_for_batch):
        if model_key == "Florence-2":
            return self._run_florence_tasks(instance, batch_images, questions_for_batch)
        elif model_key in ["BLIP", "GIT", "JoyCaption", "Moondream", "SmolVLM"]:
            return instance.infer(batch_images, questions=questions_for_batch)
        else:
            return instance.infer(batch_images)

    def _run_florence_tasks(self, instance, batch_images, questions):
        final_batch_results = [defaultdict(dict) for _ in range(len(batch_images))]
        tasks_to_run = []
        caption_style = self.config.get('florence_caption_style', 'Detailed')
        caption_map = {"Normal": "<CAPTION>", "Detailed": "<DETAILED_CAPTION>", "More Detailed": "<MORE_DETAILED_CAPTION>"}
        if caption_prompt := caption_map.get(caption_style):
            tasks_to_run.append({'type': 'caption', 'prompt': caption_prompt, 'display': f"{caption_style} Caption"})
        standard_tasks = {
            'florence_enable_od': ("<OD>", "Object Detection"),
            'florence_enable_dense_caption': ("<DENSE_REGION_CAPTION>", "Dense Caption"),
            'florence_enable_ocr': ("<OCR>", "OCR"),
            'florence_enable_ocr_with_region': ("<OCR_WITH_REGION>", "OCR w/ Region"),
            'florence_enable_region_proposal': ("<REGION_PROPOSAL>", "Region Proposal"),
        }
        for config_key, (prompt, display) in standard_tasks.items():
            if self.config.get(config_key):
                tasks_to_run.append({'type': 'standard', 'prompt': prompt, 'display': display})
        vqa_enabled = self.config.get('models_vqa_enabled', {}).get('florence', False)
        if vqa_enabled and self.config.get('florence_enable_vqa'):
            tasks_to_run.append({'type': 'vqa', 'prompts': questions, 'display': "VQA"})
        if self.config.get('florence_enable_caption_grounding'):
            tasks_to_run.append({'type': 'grounding', 'display': "Caption Grounding"})
        generated_captions = ["" for _ in range(len(batch_images))]
        for task in tasks_to_run:
            prompts_for_batch = []
            if task['type'] == 'vqa':
                prompts_for_batch = [f"<VQA>{q}" for q in questions]
            elif task['type'] == 'grounding':
                if not any(cap.strip() for cap in generated_captions):
                    self.log.emit("[Florence-2] Skipping Caption Grounding as no captions were generated.")
                    continue
                prompts_for_batch = [f"<CAPTION_TO_PHRASE_GROUNDING>{cap}" for cap in generated_captions]
            else:
                prompts_for_batch = task.get('prompts', [task.get('prompt')] * len(batch_images))
            if not any(p and p.strip() for p in prompts_for_batch): continue
            task_results = instance.infer(batch_images, prompts=prompts_for_batch)
            for i, result in enumerate(task_results):
                if not isinstance(result, dict) or "error" in result:
                    final_batch_results[i][f"{task['display']}_error"] = result or "Empty result"
                    continue
                output_value = list(result.values())[0] if len(result) == 1 else result
                if task['type'] == 'caption':
                    generated_captions[i] = output_value
                    output_key = caption_style.lower().replace(" ", "_") + "_caption"
                elif task['type'] == 'vqa': output_key = 'answer'
                elif task['type'] == 'grounding': output_key = 'caption_grounding'
                else: output_key = task['prompt'].strip("<>").lower()
                final_batch_results[i][output_key] = output_value
        return [dict(res) for res in final_batch_results]
    
    def _save_final_json(self, image_path, results_dict, question_used):
        final_json = {
            "image_path": image_path, "image_filename": os.path.basename(image_path),
            "question_used_for_image": question_used or "N/A"
        }
        txt_path = os.path.splitext(image_path)[0] + ".txt"
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                final_json["existing_caption"] = f.read().strip()
        except FileNotFoundError:
            final_json["existing_caption"] = "N/A"
        final_json.update(results_dict)
        out_dir = self.config.get('output_dir', 'output')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, os.path.splitext(os.path.basename(image_path))[0] + ".json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, indent=4, ensure_ascii=False)

    def _discover_images(self):
        formats = ('.jpg', '.jpeg', '.png', '.webp', '.avif')
        paths = []
        for root, _, files in os.walk(self.config['image_dir']):
            for f in files:
                if f.lower().endswith(formats):
                    paths.append(os.path.join(root, f))
        return paths

    def _filter_images(self, paths):
        if not self.config.get('resume_processing', True):
            return paths
        filtered_paths = []
        out_dir = self.config.get('output_dir', 'output')
        for path in paths:
            json_filename = os.path.splitext(os.path.basename(path))[0] + '.json'
            if not os.path.exists(os.path.join(out_dir, json_filename)):
                filtered_paths.append(path)
        return filtered_paths

    def _cleanup(self):
        for model_key in list(self.models.keys()):
            self.models[model_key].unload()
            del self.models[model_key]
        self.models.clear()
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()