# this is for jupyter notebooks

import torch
import gc

# Delete the model explicitly
del model
del tokenizer

# Clear PyTorch's cached memory
torch.cuda.empty_cache()

# Force Python garbage collection
gc.collect()