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

# Use 'eager' attention instead of 'flash_attention_2' for non-Ampere GPUs (T4, V100, etc.)
# Options: 'eager' (standard), 'sdpa' (scaled dot product), 'flash_attention_2' (Ampere+ only)
model = AutoModel.from_pretrained(
    model_path, 
    attn_implementation='eager', 
    trust_remote_code=True, 
    use_safetensors=True,
    local_files_only=True,
    torch_dtype=torch.bfloat16
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