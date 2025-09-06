# Sigma-Captioner
A some what optimized implementation of some light weight and popular models

This app can be used for making image text pairs to make sorting images easier or it can be used for captioning large datasets

The outputs of smol vlm will only be printed if it's outputs follow the given format 

<img width="2113" height="264" alt="image" src="https://github.com/user-attachments/assets/f8e058e5-ffc2-46e4-9b4e-73f6b1a42ab2" />

Else compute would be wasted. (CAN BE CHANGED MANUALLY)

<img width="1786" height="1345" alt="image" src="https://github.com/user-attachments/assets/2971e951-c218-4a33-a840-93abddc81024" />


Model Name	Status	
laion/CLIP-ViT-H-14-laion2B-s32B-b79K            	In Development	
laion/CLIP-ViT-L-14-laion2B-s32B-b82K	            In Development	
Salesforce/blip-vqa-base	                        Work	
Salesforce/blip-image-captioning-large	          Work	
microsoft/Florence-2-large-ft                    	Work	
microsoft/git-large-textvqa	                      Work	
HuggingFaceTB/SmolVLM-256M-Instruct              	Work	
vikhyatk/moondream2	                              In Development	
milingWolf/wd-eva02-large-tagger-v3              	Work	
fancyfeast/llama-joycaption-beta-one-hf-llava	    In Development	

To set this up follow these steps

#END OF ONLY DO THIS FOR THE FIRST TIME

python -m venv venv

cd venv && cd scripts && activate && cd.. && cd..

pip install uv

uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

uv pip install xformers==0.0.31.post1

uv pip install req.txt

#END OF ONLY DO THIS FOR THE FIRST TIME

python main.py

optionally download for better performance

flash_attn-2.7.4.post1%2Bcu128torch2.7.0cxx11abiFALSE-cp310-cp310-win_amd64.whl

pip install <location of flash attn wheel> 

<img width="551" height="314" alt="image" src="https://github.com/user-attachments/assets/4decf84b-0a13-4b35-bec2-2ceadb5ec162" />

List of models supported 

Model Name	Status	
laion/CLIP-ViT-H-14-laion2B-s32B-b79K            	In Development	
laion/CLIP-ViT-L-14-laion2B-s32B-b82K	            In Development	
Salesforce/blip-vqa-base	                        Work	
Salesforce/blip-image-captioning-large	          Work	
microsoft/Florence-2-large-ft                    	Work	
microsoft/git-large-textvqa	                      Work	
HuggingFaceTB/SmolVLM-256M-Instruct              	Work	
vikhyatk/moondream2	                              In Development	
milingWolf/wd-eva02-large-tagger-v3              	Work	
fancyfeast/llama-joycaption-beta-one-hf-llava	    In Development	
