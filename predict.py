import os
from typing import List
from cog import BasePredictor, Input, Path, File
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

"""
This predictor uses the SamModel and SamProcessor from the transformers library.
If models are not found in the cache_dir, they will be downloaded from the HuggingFace model hub.
"""

class Predictor(BasePredictor):
    def setup(self) -> None:
        cache_dir = "models"
        model_name = processor_name ="facebook/sam-vit-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = SamProcessor.from_pretrained(processor_name,cache_dir=cache_dir)
        self.model = SamModel.from_pretrained(model_name,cache_dir=cache_dir).to(self.device)
        self.model.eval()
        
    def predict(
        self,
        image: Path = Input(description="RGB Input image"),
        input_points: str = Input()
    ) -> Path:
        
        # parse input_points str:
        input_points = input_points.replace(' ', '')
        input_points = eval(input_points)
        image = str(image)
        image=Image.open(image)
        inputs = self.processor(image, input_points=[input_points], return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0]        
        mask = masks.detach().cpu().numpy()[0,0]
        mask = Image.fromarray(mask)
        mask.save("/tmp/output.png")
        return Path("/tmp/output.png")
