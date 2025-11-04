import os
import json
import glob
import argparse
from PIL import Image, ImageOps
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np

import torch
import requests
from tqdm import tqdm
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
import torchvision.transforms as T

from utils import preprocess
import re
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--adv_image_path', type=str, default='../out/test/img/')
parser.add_argument('--gen_image_path', type=str, default='../out/test/gen/img/')
parser.add_argument('--prompt_path', type=str, default='../../ImageNet-Edit_prompt.json')
args = parser.parse_args()

#model_id_or_path = "runwayml/stable-diffusion-v1-5"
model_id_or_path = "CompVis/stable-diffusion-v1-4"
pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id_or_path,
    revision="fp16", 
    torch_dtype=torch.float16,
)
def _disable_safety(images, **kwargs):
    return images, [False] * len(images)
pipe_img2img.safety_checker = _disable_safety
pipe_img2img.requires_safety_checker = False
pipe_img2img = pipe_img2img.to("cuda")
SEED = 42
STRENGTH = 0.5
GUIDANCE = 7.5
NUM_STEPS = 50

os.makedirs(args.gen_image_path, exist_ok = True)
adv_img_paths = glob.glob(args.adv_image_path+'*')
adv_img_paths = [p for p in adv_img_paths if 'png' in p or 'jpeg' in p or 'jfif' in p]

# load prompt
prompts = json.load(open(args.prompt_path))

for adv_img_path in tqdm(adv_img_paths):
    # filename for finding prompt key
    filename = os.path.basename(adv_img_path)
    print(filename)
    filename = re.search(r'(\d+)', filename).group(1) 
    print(filename)
    # load adv img
    init_image = Image.open(adv_img_path).convert("RGB")
    resize = T.transforms.Resize(512)
    center_crop = T.transforms.CenterCrop(512)
    init_image = center_crop(resize(init_image))
    print(filename)
    prompt = prompts[filename]



    with torch.autocast('cuda'):
        torch.manual_seed(SEED)
        image_gen = pipe_img2img(prompt=prompt, image=init_image, strength=STRENGTH, guidance_scale=GUIDANCE, num_inference_steps=NUM_STEPS).images[0]
    image_gen.save(f"{args.gen_image_path}{filename}_attacked.png")
    


