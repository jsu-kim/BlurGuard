import os
import numpy as np
from omegaconf import DictConfig, OmegaConf
import PIL
from PIL import Image
from einops import rearrange
import ssl
import sys
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config
from attacks import LF_PGD
import time
import wandb
import glob
import hydra
import cv2

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from utils import mp, si, cprint,load_masks_to_tensor_dict,generate_and_process_masks,save_masks_to_directory



ssl._create_default_https_context = ssl._create_unverified_context
os.environ['TORCH_HOME'] = os.getcwd()
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hub/')





def load_image_from_path(image_path: str, input_size: int) -> PIL.Image.Image:
    img = Image.open(image_path).resize((input_size, input_size),
                                        resample=PIL.Image.BICUBIC)
    return img


def load_model_from_config(config, ckpt, verbose: bool = False, device=0):

    print(f"Loading model from {ckpt}")

    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    # Support loading weight from NovelAI
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
    model.cond_stage_model.to(device)
    model.eval()
    return model



class target_model(nn.Module):
    def __init__(self, model,
                 condition: str,
                 target_info: str = None,
                 mode= 'lf', device=0):

        super().__init__()
        self.model = model
        self.condition = condition
        self.target_info = target_info
        self.mode = 'lf'

    




def init(epsilon: int = 16, steps: int = 100, 
         input_size: int = 512, object: bool = False, seed: int =23, 
         ckpt: str = None, base: str = None, mode='lf', device=0, input_prompt='a photo', noise_t=0.01, perceptual_weighting=10,eps_sigma=0.01,  sigma_weighting=10 ,learning_rate=0.001,eps_perceptual=0.01):
    if ckpt is None:
        ckpt = '../ckpt/model.ckpt'

    if base is None:
        base = '../configs/stable-diffusion/v1-inference-attack.yaml'

    imagenet_templates_small_style = ['a painting']
    
    imagenet_templates_small_object = ['a photo']

    config_path = '../configs/stable-diffusion/v1-inference-attack.yaml'
    config = OmegaConf.load(config_path)

    ckpt_path = '../ckpt/model.ckpt'
    model= load_model_from_config(config, ckpt_path, device=device).to(device)

    net = target_model(model, input_prompt, mode=mode)
    net.eval()
    

    parameters = {
        'epsilon': epsilon/255.0 * (1-(-1)),
        'steps': steps,
        'input_size': input_size,
        'mode': mode,
        'eps_sigma': eps_sigma,
        'sigma_weighting': sigma_weighting,
        'learning_rate': learning_rate
    }

    return {'net': net, 'parameters': parameters}


def infer(img: PIL.Image.Image, config, tar_img: PIL.Image.Image = None, using_target=False, device=0,masks_dict=None, filename="",  eps_sigma=0,sigma_weighting= 10,learning_rate= 0.01) -> np.ndarray:
    """
    Process the input image and generate the misted image.
    :param img: The input image or the image block to be misted.
    :param config: config for the attack.
    :param img: The target image or the target block as the reference for the textural loss.
    :returns: A misted image.
    """

    net = config['net']
    net=net
    parameters = config['parameters']
    mode = parameters['mode']
    epsilon = parameters["epsilon"]
    cprint(f'epsilon: {epsilon}', 'y')
    steps = parameters["steps"]
    input_size = parameters["input_size"]
    eps_sigma=parameters["eps_sigma"]
    sigma_weighting=parameters["sigma_weighting"]
    learning_rate=parameters["learning_rate"]

    
  


    img = img.convert('RGB')
    img = np.array(img).astype(np.float32) / 127.5 - 1.0
    img = img[:, :, :3]


    trans = transforms.Compose([transforms.ToTensor()])
    
    data_source = torch.zeros([1, 3, input_size, input_size]).to(device)
    tar_img_tensor= torch.zeros([1, 3, input_size, input_size]).to(device)
    
    data_source[0] = trans(img).to(device)


    net.target_info = tar_img_tensor 

    net.mode = mode

    label = torch.zeros(data_source.shape).to(device)

    
    time_start_attack = time.time()

    attack = LF_PGD(net, epsilon, steps ,targeted=True,masks_dict=masks_dict, filename=filename,  eps_sigma=eps_sigma,sigma_weighting=sigma_weighting,learning_rate=learning_rate,attack_type=mode)
    attack_output,advL, percL, sigmaL, final_sigmas = attack.perturb(data_source, label)
    print('Attack takes: ', time.time() - time_start_attack)
 
 
    
    output = attack_output[0]
    save_adv = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0).detach()
    grid_adv = 255. * rearrange(save_adv, 'c h w -> h w c').cpu().numpy()
    grid_adv = grid_adv
    return grid_adv,advL, percL, sigmaL, final_sigmas





@hydra.main(version_base=None, config_path="../configs/attack", config_name="base")
def main(cfg : DictConfig):
    

    
    print(cfg.attack)
    time_start = time.time()
    args = cfg.attack
    epsilon = args.epsilon
    steps = args.steps
    input_size = args.input_size
    mode=args.mode
    mode_name = args.mode
    eps_sigma=args.eps_sigma
    sigma_weighting=args.sigma_weighting
    learning_rate=args.learning_rate
    input_prompt=args.input_prompt
    
    run_name = "BlurGuard"
        
    output_path, img_path,results_dir = args.output_path, args.img_path,args.results_path
    using_target = args.using_target
    device=args.device

    config = init(epsilon=epsilon, steps=steps, 
                  mode=mode, device=device, 
                  input_prompt=input_prompt,
                 eps_sigma=eps_sigma,
                 sigma_weighting=sigma_weighting,
                  learning_rate=learning_rate
                 )

    img_paths = glob.glob(img_path+'/*.png')


    sam_checkpoint = "./segment_anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam_model.to(device=device)

    all_logs = []

    run_dir = os.path.join(output_path, run_name)
    os.makedirs(run_dir, exist_ok=True)

    img_out_dir = os.path.join(run_dir, "img")
    os.makedirs(img_out_dir, exist_ok=True)

    os.makedirs(results_dir, exist_ok=True)
    results_json_path = os.path.join(results_dir, f"{run_name}.json")
    
    
    for image_path in tqdm(img_paths):
        file_name = os.path.splitext(os.path.basename(image_path))[0]

        
        
        # create mask
        mask_dir = f"../mask/{file_name}/"
        mp(mask_dir)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks_dict = generate_and_process_masks(sam_model, image, device,points_per_side=2)
  
        print(f"Number of masks stored: {len(masks_dict)}")
 
        
        img = load_image_from_path(image_path, input_size)

        output_image = np.zeros([input_size, input_size, 3])
        img = Image.fromarray(np.array(img))
        
        

        output_image,advL, percL, sigmaL, final_sigmas = infer(img, config, using_target=using_target, device=device,masks_dict=masks_dict,  filename=file_name,  eps_sigma=eps_sigma,sigma_weighting=sigma_weighting, learning_rate=learning_rate)
        
        
        log_dict = {
        "filename": file_name,
        "adv_loss": advL,
        "perceptual_loss": percL,
        "sigma_loss": sigmaL,
        "final_sigmas": final_sigmas
        }
        all_logs.append(log_dict)
        with open(results_json_path, "w") as f:
            json.dump(all_logs, f, indent=4)
        
        output = Image.fromarray(output_image.astype(np.uint8))
        
        time_start_sdedit = time.time()

        print('SDEdit takes: ', time.time() - time_start_sdedit)
        
        
        output_name = img_out_dir + f'/{file_name}_attacked.png'
        output.save(output_name)

        
        print('TIME CMD=', time.time() - time_start)


if __name__ == '__main__':
    main()