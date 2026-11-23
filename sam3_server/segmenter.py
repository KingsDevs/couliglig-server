"""SAM3 (Hugging Face transformers) text-prompt segmentation.

The only module that imports torch/transformers. facebook/sam3 is gated: accept
its terms on huggingface.co and provide HF_TOKEN (or `huggingface-cli login`).
"""

import logging

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

logger = logging.getLogger(__name__)


class Sam3Segmenter:
    """Callable: (rgb (H, W, 3) uint8, text) -> (mask uint8 (H, W), score, n_instances)
    for the best-scoring instance, or None when nothing matches."""

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: str = "cuda",
        fp16: bool = True,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> None:
        self._device = device
        self._fp16 = fp16
        self._dtype = torch.float16 if fp16 else torch.float32
        self._threshold = threshold
        self._mask_threshold = mask_threshold
        self.info = {
            "model_id": model_id,
            "device": device,
            "fp16": fp16,
            "threshold": threshold,
            "mask_threshold": mask_threshold,
        }

        logger.info("Loading SAM3 from %s (fp16=%s, device=%s) ...", model_id, fp16, device)
        self._model = (
            Sam3Model.from_pretrained(model_id, torch_dtype=self._dtype)
            .to(device)
            .eval()
        )
        self._processor = Sam3Processor.from_pretrained(model_id)
        logger.info("SAM3 loaded.")

    def __call__(self, rgb: np.ndarray, text: str):
        with torch.inference_mode():
            inputs = self._processor(
                images=Image.fromarray(rgb), text=text, return_tensors="pt"
            ).to(self._device)
            # Float tensors (pixel_values) follow the model dtype; integer ones
            # (input_ids, attention masks) are left alone by BatchFeature.to.
            if self._fp16:
                inputs = inputs.to(self._dtype)
            outputs = self._model(**inputs)
            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=self._threshold,
                mask_threshold=self._mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

        if len(results["scores"]) == 0:
            return None
        best = int(results["scores"].argmax())
        mask = results["masks"][best].cpu().numpy().astype(np.uint8)
        return mask, float(results["scores"][best]), int(len(results["scores"]))
