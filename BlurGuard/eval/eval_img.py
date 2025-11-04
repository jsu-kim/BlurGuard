# Our code is based on Diff-Protect repository: https://github.com/xavihart/Diff-Protect (Xue, 2023).

import os
import numpy as np
from omegaconf import DictConfig, OmegaConf
import PIL
from PIL import Image
from einops import rearrange
import ssl
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
import time
import wandb
import glob
import hydra
from utils import mp, si, cprint, load_png
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
from utils import lpips_, ssim_, psnr_, vifp_, clip_, fid_,image_reward_
import argparse

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['TORCH_HOME'] = os.getcwd()
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hub/')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description="Evaluate configurations")
parser.add_argument("--path", type=str, required=True, help="Full path to the directory to evaluate")
parser.add_argument("--source", type=str, required=True, help="source gen and img path")

args = parser.parse_args()


input_path = args.path
source_path = args.source


base_path, last_dir = os.path.split(input_path.rstrip("/"))



def load_image_from_path(image_path: str, input_size: int) -> PIL.Image.Image:
    """
    Load image form the path and reshape in the input size.
    :param image_path: Path of the input image
    :param input_size: The requested size in int.
    :returns: An :py:class:`~PIL.Image.Image` object.
    """
    img = Image.open(image_path).resize((input_size, input_size),
                                        resample=PIL.Image.BICUBIC)
    return img


def load_model_from_config(config, ckpt, verbose: bool = False):
    """
    Load model from the config and the ckpt path.
    :param config: Path of the config of the SDM model.
    :param ckpt: Path of the weight of the SDM model
    :param verbose: Whether to show the unused parameters weight.
    :returns: A SDM model.
    """
    print(f"Loading model from {ckpt}")

    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    if "state_dict" in sd:
        import copy
        sd_copy = copy.deepcopy(sd)
        for key in sd.keys():
            if key.startswith('cond_stage_model.transformer') and not key.startswith('cond_stage_model.transformer.text_model'):
                newkey = key.replace('cond_stage_model.transformer', 'cond_stage_model.transformer.text_model', 1)
                sd_copy[newkey] = sd[key]
                del sd_copy[key]
        sd = sd_copy

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model

def normalized_(x):
    return (x - x.min() / x.max() - x.min())


def get_dir_name_from_config(mode,prefix=base_path):
    mode_name = mode
    dir_name = f'{prefix}/{mode_name}/'
    return dir_name

EXP_LIST = [
    (last_dir)

]

STYLE_LIST = [
    'img',
    'gen'
]

MIMICRY_LIST=[
    'img',
    'jpeg',
    'jpeg_ups',
    'grid_pure',
    'diff_pure',
    'noise_ups',
    'impress',
    'pdm_pure', 
]


def dm_range(x):
    return 2 * x - 1

def rdm_range(x):
    return (x+1)/2

def encode(x, model):
    z = model.get_first_stage_encoding(model.encode_first_stage(x)).to(x.device)
    return z

def linf(x):
    return torch.abs(x).max()

@torch.no_grad()
def main():
    
    
    
    for exp_config in tqdm(EXP_LIST):
        cprint(exp_config, 'y')

        mode= exp_config
        dir_name = get_dir_name_from_config(mode)
        cprint('fetching dir: ' + dir_name, 'g')
        
        clean_name = get_dir_name_from_config('none/img',prefix=source_path)
        clean_gen_name = get_dir_name_from_config('none/gen',prefix=source_path)
        input_path
        save_path= os.path.join(input_path, 'out_stat')

        mp(save_path)

        x_clean_source_path_list=[]
        for style in STYLE_LIST:
            if style == 'img':

                
                linf_z_list = []
                linf_x_list = []

                ssim_list = []
                lpips_list = []
                psnr_list = []



                z_max = 0

                x_list = []
                x_adv_list =[]
                x_adv_path_list=[]
                x_clean_source_path_list=[]
                file_list = sorted([f for f in os.listdir(os.path.join(dir_name, style)) if f.endswith('_attacked.png')],key=lambda x: int(x.split('_')[0]))

                img_dir=os.path.join(dir_name, style)

                for file_name in tqdm(file_list):
                    img_path = os.path.join(dir_name, style, file_name)
                    clean_img_path = os.path.join(clean_name, file_name)
                    x_adv_path_list.append(img_path)
                    x_clean_source_path_list.append(clean_img_path)
                    if not os.path.exists(img_path):
                        print(f"🚨 Missing file: img_path does not exist -> {img_path}")
                        continue
                    if not os.path.exists(clean_img_path):
                        print(f"🚨 Missing file: clean_img_path does not exist -> {clean_img_path}")
                        continue

                    x_adv = load_png(img_path, 512)[None, ...].to(device)
                    x = load_png(clean_img_path, 512)[None, ...].to(device)

                    if x_adv is None:
                        print(f"⚠️ Warning: x_adv is None for {img_path}")
                        continue
                    if x is None:
                        print(f"⚠️ Warning: x is None for {clean_img_path}")
                        continue


                    if x_adv.shape != x.shape:
                        print(f"❌ Dimension Mismatch! Skipping {file_name}")
                        continue

                    x, x_adv = dm_range(x), dm_range(x_adv)

                    ssim_x = ssim_(img_path, clean_img_path)
                    psnr_x = psnr_(img_path, clean_img_path)

                    ssim_list.append(ssim_x)
                    psnr_list.append(psnr_x)
                    x_list.append(x)
                    x_adv_list.append(x_adv)
                    

                save_path_style = os.path.join(save_path, style+'/')
                mp(save_path_style)

                if len(x_adv_list) == 0 or len(x_list) == 0:
                    print(f"⚠️ Warning: No valid images found for {style}")
                    continue
                
                x_adv_all = torch.cat(x_adv_list, 0)
                x_all = torch.cat(x_list, 0)

                ia, clip_dir, clean_clip, adv_clip = clip_(x_adv_path_list, x_clean_source_path_list)
                ia = ia.diag().tolist()
                clean_clip = clean_clip.diag().tolist()
                adv_clip = adv_clip.diag().tolist()
                lpips_score = lpips_(x_all, x_adv_all)
                
                lpips_score = lpips_score[:, 0, 0, 0].cpu().tolist()
                
                
                clean_img_dir = os.path.join(clean_name)
                fid_x = fid_(img_dir, clean_img_dir)
                fid_x_list = [fid_x]
  
                torch.save({
                    'ssim':ssim_list,
                    'lpips':lpips_score,
                    'psnr':psnr_list,
                    'ia' :ia,
                    'adv_clip' :adv_clip,
                    'fid' : fid_x_list
                }, save_path_style+'/x_adv_metrics.bin')


            elif style == 'gen':
                for mimicry in MIMICRY_LIST:

                    linf_z_list = []
                    linf_x_list = []

                    ssim_list = []
                    lpips_list = []
                    psnr_list = []

                    z_max = 0

                    x_list = []
                    x_adv_list =[]
                    x_adv_path_list=[]
                    x_clean_path_list=[]
                    

                    file_list = sorted(
                                            [f for f in os.listdir(os.path.join(dir_name, style, mimicry)) if f.endswith(('_attacked.png', '_attacked.jpeg', '_attacked.jpg'))],
                                            key=lambda x: int(x.split('_')[0])
                                        )

                    target_dir = os.path.join(dir_name, style,mimicry)
                    print(target_dir)

                    for file_name in tqdm(file_list):
                        img_path = os.path.join(dir_name, style, mimicry, file_name)
                        
                        clean_file_name = file_name.replace('.jpeg', '.jpg').replace('.png', '.png')
                        clean_img_path = os.path.join(clean_gen_name, clean_file_name)
                        img_dir= os.path.join(dir_name,  f"{style}/{mimicry}")
                        clean_gen_dir=os.path.join(clean_gen_name)
                        
                        if not os.path.exists(img_path):
                            print("NO SUCH PATH", img_path)
                            break
                        x_adv_path_list.append(img_path)
                        x_clean_path_list.append(clean_img_path)
                        x_adv = load_png(img_path, 512)[None, ...].to(device)
                        x     = load_png(clean_img_path, 512)[None, ...].to(device)

                        x, x_adv = dm_range(x), dm_range(x_adv)

                        x_list.append(x)
                        x_adv_list.append(x_adv)

                        ssim_x = ssim_(img_path, clean_img_path)
                        psnr_x = psnr_(img_path, clean_img_path)

                        ssim_list.append(ssim_x)
                        psnr_list.append(psnr_x)



                    save_path_style_mimicry  = os.path.join(save_path, f"{style}/{mimicry}/")
                    mp(save_path_style_mimicry)
                    
                    x_adv_all = torch.cat(x_adv_list, 0)
                    x_all = torch.cat(x_list, 0)
                    lpips_score = lpips_(x_all, x_adv_all)
                    lpips_score = lpips_score[:, 0, 0, 0].cpu().tolist()
                    fid_x=fid_(img_dir,clean_gen_dir)
                    fid_x_list = [fid_x]

                    ia,clip_dir,clean_clip,adv_clip = clip_(x_adv_path_list,x_clean_path_list,x_clean_source_path_list)

                    ia=ia.diag().tolist()
                    clean_clip=clean_clip.diag().tolist()
                    adv_clip=adv_clip.diag().tolist()
                    

                    
                    torch.save({
                        'ssim':ssim_list,
                        'lpips':lpips_score,
                        'psnr':psnr_list,
                        'ia' :ia,
                        'adv_clip': adv_clip,
                        'fid' : fid_x_list
                    }, save_path_style_mimicry +'/x_adv_metrics.bin')



if __name__ == '__main__':
    main()
    
