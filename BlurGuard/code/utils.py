import torch
import numpy as np
import torchvision
from colorama import Fore, Back, Style
import os
import torchvision.transforms as transforms
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch_dct as dct
import glob
import cv2


def cprint(x, c):
    c_t = ""
    if c == 'r':
        c_t = Fore.RED
    elif c == 'g':
        c_t = Fore.GREEN
    elif c == 'y':
        c_t = Fore.YELLOW
    elif c == 'b':
        c_t = Fore.BLUE
    print(c_t, x)
    print(Style.RESET_ALL)

def si(x, p, to_01=False, normalize=False):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    
    if x.dtype == torch.float16:
        x = x.float()

    if to_01:
        torchvision.utils.save_image((x + 1) / 2, p, normalize=normalize)
    else:
        torchvision.utils.save_image(x, p, normalize=normalize)


def mp(p):

    first_dot = p.find('.')
    last_slash = p.rfind('/')
    if first_dot < last_slash:
        assert ValueError('Input path seems like a/b.c/d/g, which is not allowed :(')
    p_new = p[:last_slash] + '/'
    if not os.path.exists(p_new):
        os.makedirs(p_new)

def generate_and_process_masks(model, image, device, points_per_side, pred_iou_thresh=0.86, stability_score_thresh=0.92, 
                               crop_n_layers=1, crop_n_points_downscale_factor=2, min_mask_region_area=100):

    mask_generator = SamAutomaticMaskGenerator(
        model=model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area
    )
    masks = mask_generator.generate(image)

    masks_dict = {}
    sorted_masks = sorted(masks, key=lambda x: x['area'])
    used_mask = np.zeros_like(masks[0]['segmentation'], dtype=np.uint8)

    for i, mask_dict in enumerate(sorted_masks):
        segmentation = mask_dict['segmentation']
        mask_array = np.where(segmentation, 1, 0).astype(np.uint8)
        mask_array = np.where(used_mask == 0, mask_array, 0)
        used_mask = np.where(mask_array > 0, 1, used_mask)
        mask_tensor = torch.tensor(mask_array, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
        masks_dict[f'mask{i+1}'] = mask_tensor

    unused_mask = np.where(used_mask == 0, 1, 0).astype(np.uint8)
    if np.any(unused_mask):
        unused_mask_tensor = torch.tensor(unused_mask, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
        masks_dict[f'mask{len(masks_dict) + 1}'] = unused_mask_tensor
    del mask_generator
    return masks_dict

def save_masks_to_directory(masks_dict, directory="../out/mask"):

    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, (mask_name, mask_tensor) in enumerate(masks_dict.items()):
        mask_array = mask_tensor.squeeze().cpu().numpy() * 255  
        mask_image = Image.fromarray(mask_array.astype(np.uint8))
        mask_image.save(os.path.join(directory, f"{mask_name}.png"))


def load_masks_to_tensor_dict(directory, device="cuda"):

    mask_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    masks_dict = {}

    for i, mask_file in enumerate(mask_files):
        mask_path = os.path.join(directory, mask_file)
        mask_image = Image.open(mask_path)
        mask_array = np.array(mask_image)  
        

        mask_array = (mask_array > 0).astype(np.uint8)
        

        mask_tensor = torch.tensor(mask_array, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
        

        masks_dict[f'mask{i+1}'] = mask_tensor

    return masks_dict




totensor = transforms.ToTensor()
topil = transforms.ToPILImage()

def recover_image(image, init_image, mask, background=False):
    image = totensor(image)
    mask = totensor(mask)
    init_image = totensor(init_image)
    if background:
        result = mask * init_image + (1 - mask) * image
    else:
        result = mask * image + (1 - mask) * init_image
    return topil(result)

def load_png(p, size, mode='bicubic'):
    x = Image.open(p).convert('RGB')

    if mode == 'bicubic':
        inter_mode = transforms.InterpolationMode.BICUBIC
    elif mode == 'bilinear':
        inter_mode = transforms.InterpolationMode.BILINEAR


    if size is not None:
        transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=inter_mode),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        

    x = transform(x)
    return x



def get_plt_color_list():
    return ['red', 'green', 'blue', 'black', 'orange', 'yellow', 'black']


    
   
def draw_bound(a, m, color):
    if a.device != 'cpu':
        a = a.cpu()
    if color == 'red':
        c = torch.ones((3, 224, 224)) * torch.tensor([1, 0, 0])[:, None, None]
    if color == 'green':
        c = torch.ones((3, 224, 224)) * torch.tensor([0, 1, 0])[:, None, None]
    
    return c * m + a * (1 - m)



def smooth_loss(output, weight):
    tv_loss = torch.sum(
        (output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) * (output[:, :-1, :-1, :] - output[:, :-1, 1:, :]) + \
        (output[:, :-1, :-1, :] - output[:, 1:, :-1, :]) * (output[:, :-1, :-1, :] - output[:, 1:, :-1, :])) / 2.0
    return tv_loss * weight



def compose_images_in_folder(p, dim, size=224):
    l = glob.glob(p + '*.png')
    l += glob.glob(p + '*.jpg')
    print(l)
    return torch.cat([load_png(item, size) for item in l], dim)



def get_bkg(m, e=0.01):
    assert  len(m.shape) == 4
    b = [0.2667, 0, 0.3255]
    m_0 = (m[:, 0, ...] > b[0] - e) * (m[:, 0, ...] < b[0] + e)
    m_1 = (m[:, 1, ...] > b[1] - e) * (m[:, 1, ...] < b[1] + e)
    m_2 = (m[:, 2, ...] > b[2] - e) * (m[:, 2, ...] < b[2] + e)
    m =   1. - (m_0 * m_1 * m_2).float()
    return m[None, ...]

