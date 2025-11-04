import os
from PIL import Image, ImageOps
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import requests
from tqdm import tqdm
from io import BytesIO
import torchvision.transforms as T
from typing import Union, List, Optional, Callable
import argparse
from utils import preprocess, prepare_mask_and_masked_image, recover_image, prepare_image
to_pil = T.ToPILImage()
from diffusers import DiffusionPipeline
from impress import impress
from diffusers import DiffusionPipeline
import random


import time
from tqdm import tqdm

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def main(args):
    pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe = pipe.to(args.device)
    for name, param in pipe.vae.named_parameters():
        param.requires_grad = False

    torch.manual_seed(args.manual_seed)

    adv_dir = args.adv_dir
    save_dir_pur = adv_dir.replace('/img', '/impress')
    os.makedirs(save_dir_pur, exist_ok=True)

    file_dir_list = sorted([f for f in os.listdir(adv_dir) if f.endswith(".png") and ".ipynb" not in f])
    image_file_names = [os.path.basename(path)[:-4] for path in file_dir_list]

    processed_files = {os.path.splitext(f)[0] for f in os.listdir(save_dir_pur) if f.endswith(".png")}
    unprocessed_files = [name for name in image_file_names if name not in processed_files]

    total_files = len(unprocessed_files)
    print(f"Total files to process: {total_files}")

    start_time = time.time()
    for i, image_name in enumerate(tqdm(unprocessed_files, desc="Processing files", unit="file")):
        adv_image = Image.open(os.path.join(adv_dir, image_name + '.png')).convert('RGB').resize((512, 512))
        x_adv = preprocess(adv_image).to(args.device).half()

        x_purified = impress(
            x_adv,
            model=pipe.vae,
            clamp_min=-1,
            clamp_max=1,
            eps=args.pur_eps,
            iters=args.pur_iters,
            lr=args.pur_lr,
            pur_alpha=args.pur_alpha,
            noise=args.pur_noise,
        )
        x_purified = (x_purified / 2 + 0.5).clamp(0, 1)
        purified_image = to_pil(x_purified[0]).convert("RGB")
        purified_image.save(f"{save_dir_pur}/{image_name}.png")

        elapsed_time = time.time() - start_time
        avg_time_per_file = elapsed_time / (i + 1)
        remaining_files = total_files - (i + 1)
        estimated_time_left = avg_time_per_file * remaining_files
        tqdm.write(f"Remaining time: {estimated_time_left:.2f}s")

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='diffusion attack')

    parser.add_argument('--model', default="runwayml/stable-diffusion-inpainting", type=str,
                        help='stable diffusion weight')

    parser.add_argument('--adv_dir', default="../out/BlurGuard/img", type=str, help='l2 linf')



    parser.add_argument('--neg_feed', type=float, default=-1.)
    parser.add_argument('--pur_eps', default=0.1, type=float, help='ae Hyperparameters')
    parser.add_argument('--pur_iters', default=1000, type=int, help='ae Hyperparameters')
    parser.add_argument('--pur_lr', default=0.005, type=float, help='ae Hyperparameters')
    parser.add_argument('--pur_alpha', default=0.01, type=float, help='ae Hyperparameters')
    parser.add_argument('--pur_noise', default=0.05, type=float, help='ae Hyperparameters')

    parser.add_argument('--guidance', default=7.5, type=float, help='learning rate.')
    parser.add_argument('--diff_steps', default=4, type=int, help='learning rate.')

    parser.add_argument('--batch_size', default=16, type=int, help='batch size.')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate.')
    parser.add_argument('--epoch', default=50, type=int, help='training epoch.')

    parser.add_argument('-c', '--checkpoint', default='./ckpt', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--manual_seed', default=42, type=int, help='manual seed')

    parser.add_argument('--device', default='cuda:0', type=str,
                        help='device used for training')

    args = parser.parse_args()
    np.random.seed(seed = args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.backends.cudnn.deterministic=True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sys.path.append("..")
    main(args)
