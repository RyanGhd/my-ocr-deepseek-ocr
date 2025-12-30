# DeepSeek OCR - Offline Mode
from transformers import AutoModel, AutoTokenizer
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Use local path instead of HuggingFace model name
model_path = '/opt/models/deepseek-ocr'

# Set offline mode environment variables
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Load from local path with local_files_only=True
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModel.from_pretrained(
    model_path, 
    _attn_implementation='flash_attention_2', 
    trust_remote_code=True, 
    use_safetensors=True,
    local_files_only=True
)

model = model.eval().cuda().to(torch.bfloat16)

prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'images/1.png'
output_path = 'images_output'

res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path, 
    base_size=1024, 
    image_size=640, 
    crop_mode=True, 
    save_results=True, 
    test_compress=True
)