from transformers import SamModel, SamProcessor
import torch

processor_name = "facebook/sam-vit-base"
cache_dir = "models"
model_name ="facebook/sam-vit-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = SamProcessor.from_pretrained(processor_name,cache_dir=cache_dir)
model = SamModel.from_pretrained(model_name,cache_dir=cache_dir).to(device)