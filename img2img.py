import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

print("Loading model...")

pipe = AutoPipelineForImage2Image.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    safety_checker=None
)

pipe.enable_model_cpu_offload()

input_image = Image.open("inputs/input.jpg").convert("RGB")

prompt = input("Masukan prompt anda: ")
generator = torch.Generator(device="cuda").manual_seed(42)

print("Mengubah Gambar ...")

image = pipe(
    prompt=prompt,
    image=input_image,
    strength=0.4,
    num_inference_steps=20,
    height=256,
    width=256,
    generator=generator
).images[0]

image.save("outputs/img2img_result.png")

print("Gambar berhasil disimpan di outputs/img2img_result.png")