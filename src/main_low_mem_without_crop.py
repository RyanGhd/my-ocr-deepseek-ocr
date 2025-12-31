# DeepSeek OCR - Offline Mode (Memory Optimized for T4 GPU)
from transformers import AutoModel, AutoTokenizer
import torch
import gc
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Use local path instead of HuggingFace model name
model_path = '/opt/models/deepseek-ocr'

# Set offline mode environment variables
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Clear any existing GPU memory
torch.cuda.empty_cache()
gc.collect()

# Load from local path with local_files_only=True
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    trust_remote_code=True,
    local_files_only=True
)

# Use 'eager' attention instead of 'flash_attention_2' for non-Ampere GPUs (T4, V100, etc.)
# Options: 'eager' (standard), 'sdpa' (scaled dot product), 'flash_attention_2' (Ampere+ only)
# Note: Using low_cpu_mem_usage=True to reduce peak memory during loading
model = AutoModel.from_pretrained(
    model_path, 
    attn_implementation='eager', 
    trust_remote_code=True, 
    use_safetensors=True,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)

model = model.eval().cuda().to(torch.bfloat16)

# Clear cache after model loading
torch.cuda.empty_cache()
gc.collect()

prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'images/1.png'
output_path = 'images_output'

# Use smaller image sizes to reduce memory consumption
# Reduced base_size from 1024 to 768, image_size from 640 to 448
# Also reduced max_new_tokens if possible (via inference_mode)
with torch.inference_mode():
    res = model.infer(
        tokenizer, 
        prompt=prompt, 
        image_file=image_file, 
        output_path=output_path, 
        base_size=768,      # Reduced from 1024 to save memory
        image_size=448,     # Reduced from 640 to save memory
        crop_mode=False,    # Disable crop mode to reduce memory (processes fewer patches)
        save_results=True, 
        test_compress=True
    )