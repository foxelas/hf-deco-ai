access_token = ""
with open("access_token.txt", "r") as f:
    access_token = f.read()

import torch
from diffusers import StableDiffusionImg2ImgPipeline
model_path = "CompVis/stable-diffusion-v1-4"
import requests
from io import BytesIO
from PIL import Image

run_on_cpu = not(torch.cuda.is_available())
if run_on_cpu:
    device = torch.device("cpu")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        use_auth_token=access_token
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

else:
    device = torch.device("cuda")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=access_token
    )
    pipe = pipe.to(device)
    #pipe.enable_attention_slicing()

#######################################################
## Input and Parameters
target_url = "https://t4.ftcdn.net/jpg/03/46/82/87/360_F_346828778_LM1vQWt1RLsrlHZMeNYeHnBBP90APQab.jpg"
prompt = "A living room decorated in boho style with lots of furniture."
strength = 0.75
guidance_scale = 7.5

#######################################################
response = requests.get(target_url)
init_img = Image.open(BytesIO(response.content)).convert("RGB")
init_img = init_img.resize((768, 512))
init_img.save("input.jpg")
print("Saved input image as [input.jpg] in local directory.")

generator = torch.Generator(device=device).manual_seed(1024)
if run_on_cpu:
    image = pipe(prompt=prompt, init_image=init_img, strength=strength, guidance_scale=guidance_scale, generator=generator).images[0]
else:
    with torch.autocast("cuda"):
        image = pipe(prompt=prompt, init_image=init_img, strength=strength, guidance_scale=guidance_scale, generator=generator).images[0]

image.save("output.jpg")

print("Saved output image as [output.jpg] in local directory.")

