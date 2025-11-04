import os
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--adv_image_path', type=str)
parser.add_argument('--out_image_path', type=str)
parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

image_folder = args.adv_image_path
output_folder = args.out_image_path
seed = args.seed  

model_id = "stabilityai/stable-diffusion-x4-upscaler"
prompt = "a photo"

os.makedirs(output_folder, exist_ok=True)

pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)

pipeline.enable_xformers_memory_efficient_attention()
pipeline = pipeline.to("cuda")

if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Please ensure that your system has a working GPU setup.")

if not image_folder.endswith('/'):
    image_folder += '/'

image_paths = glob.glob(os.path.join(image_folder, '*'))
image_paths = [p for p in image_paths if 'attacked' in p and (p.endswith('.jpeg') or p.endswith('.jpg') or p.endswith('.png'))]

print(f"Processing images in: {image_folder}")
for image_path in image_paths:
    if "noise/" in image_path:
        output_path = image_path.replace("noise/", "noise_ups/")
    elif "jpeg/" in image_path:
        output_path = image_path.replace("jpeg/", "jpeg_ups/")
    
    if not (output_path.endswith('.jpeg') or output_path.endswith('.jpg') or output_path.endswith('.png')):
        continue

    adv_image = Image.open(image_path)
    adv_image = adv_image.resize((512, 512))

    generator = torch.Generator(device="cuda").manual_seed(seed)

    upscaled_image = pipeline(prompt=prompt, image=adv_image, generator=generator).images[0]

    upscaled_image.resize((512, 512)).save(output_path)
    
    print(f"Image {image_path} has been upscaled and saved to {output_path}")

    adv_image.close()
    upscaled_image.close()
    del adv_image, upscaled_image
    torch.cuda.empty_cache()
