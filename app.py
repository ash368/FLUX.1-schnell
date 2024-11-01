from diffusers import FluxPipeline
import torch
from io import BytesIO
import base64

class InferlessPythonModel:
    def initialize(self):
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")

    def infer(self, inputs):
        prompt = inputs["prompt"]
        height = inputs.get("height", 1024)
        width = inputs.get("width", 1024)
        guidance_scale = inputs.get("guidance_scale", 3.5)
        num_inference_steps = inputs.get("num_inference_steps", 25)
        # max_sequence_length = inputs.get("max_sequence_length", 256)

        image = self.pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            
        ).images[0]

        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode()

        return {"generated_image_base64": img_str}

    def finalize(self):
        self.pipe = None
