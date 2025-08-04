from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
from typing import List, Dict
import numpy as np
import cv2


class YolosDetector:
    def __init__(
        self, model_name: str = "hustvl/yolos-tiny", dtype=torch.float32, use_cuda=True
    ):
        self.dtype = dtype

        self.image_processor = YolosImageProcessor.from_pretrained(model_name)
        self.model = YolosForObjectDetection.from_pretrained(
            model_name,
            use_safetensors=True,
            attn_implementation="sdpa",
            torch_dtype=self.dtype,
        )

        if use_cuda:
            self.model = self.model.cuda()

    @torch.inference_mode()
    def __call__(self, image: np.ndarray, from_bgr: bool = False):
        if from_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        inputs = self.image_processor(images=image, return_tensors="pt").to(
            device=self.model.device, dtype=self.dtype
        )
        outputs = self.model(**inputs)

        H, W, _ = image.shape

        target_sizes = torch.tensor([[H, W]])
        results: List[Dict[str, torch.Tensor]] = (
            self.image_processor.post_process_object_detection(
                outputs=outputs,
                threshold=0.9,
                target_sizes=target_sizes,
            )
        )

        img_result = results[0]

        # scores: torch.Tensor = img_result["scores"]
        labels: torch.Tensor = img_result["labels"]
        boxes: torch.Tensor = img_result["boxes"]

        return boxes[labels == 1]


if __name__ == "__main__":
    url = "outputs/hand-people-girl-woman-white-female-607482-pxhere.com.jpg"
    image = Image.open(url)

    detector = YolosDetector(model_name="hustvl/yolos-tiny", dtype=torch.float32)
    boxes = detector(np.asarray(image))

    print(boxes)
