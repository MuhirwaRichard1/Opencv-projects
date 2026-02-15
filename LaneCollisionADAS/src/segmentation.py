
import cv2
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class LaneSegmenter:
    def __init__(self, model_id="nvidia/segformer-b0-finetuned-cityscapes-512-1024"):
        print(f"Loading SegFormer model: {model_id}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.processor = SegformerImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def segment_road(self, frame):
        """
        Returns a binary mask where 1 = road/lane, 0 = background.
        """
        # SegFormer expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        inputs = self.processor(images=rgb_frame, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Post-process logits
        # Interpolate to original size
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=frame.shape[:2], # (H, W)
            mode="bilinear",
            align_corners=False,
        )
        
        # Argmax to get labels
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        
        # Cityscapes: Index 0 is 'road'
        # Move to CPU numpy
        pred_seg = pred_seg.cpu().numpy().astype(np.uint8)
        
        # Create binary mask for road (label 0)
        road_mask = (pred_seg == 0).astype(np.uint8) * 255
        
        return road_mask
