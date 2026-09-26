import torch 
from diffusers import DiffusionPipeline

MODEL_ID = "segmind/tiny-sd"

print ("Loading model...")

pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")

prompt = input("Masukan prompt anda: ")

print ("Menghasilkan Gambar ...")

image = pipe(
    prompt,
    height=384,
    width=384
).images[0]

image.save("outputs/result.png")

print ("Gambar berhasil dihasilkan dan disimpan di outputs/result.png")