from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
import torch
from PIL import Image
import matplotlib.pyplot as plt

print(torch.cuda.is_available())

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
)
pipe.to("cuda")

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

# Display the image
plt.imshow(image)
plt.axis("off")  # Hide the axes for a better view
plt.show()
