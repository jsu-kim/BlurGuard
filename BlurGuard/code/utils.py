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
from skimage.metrics import structural_similarity as ssim

from sewar.full_ref import vifp

from transformers import CLIPModel, CLIPProcessor
import subprocess
import ImageReward as reward
import shutil

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
    
    # FP16 텐서를 FP32로 변환
    if x.dtype == torch.float16:
        x = x.float()

    if to_01:
        torchvision.utils.save_image((x + 1) / 2, p, normalize=normalize)
    else:
        torchvision.utils.save_image(x, p, normalize=normalize)


def mp(p):
    # if p is like a/b/c/d.png, then only make a/b/c/
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




def cal_mse(ratio, grad,editor,X,masks_dict,noise_t,file_name):
    num_masks = len(masks_dict)
    masked_grad = torch.zeros_like(X, device='cuda')
    mse_losses = torch.zeros(num_masks, device='cuda')  
    
    loss=0
    save_dir = f"../mask_img/{file_name}/"  
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  
    for i in range(num_masks):
        mask = masks_dict[f'mask{i+1}']
        mask_np = mask.squeeze().cpu().numpy() * 255  
        mask_image = Image.fromarray(mask_np.astype('uint8'))  
        name = f"mask_{i}.png"
        save_path = os.path.join(save_dir,name)
        mask_image.save(save_path)
        part_grad = zero_low_frequencies(grad, ratio[i])*mask 
        masked_grad+=part_grad 

    edit_one_step = editor.edit_list(X+masked_grad  , restep=None, t_list=[noise_t]).cuda()

    left_image = edit_one_step[:, :, :, :edit_one_step.shape[3] // 2] 
    right_image = edit_one_step[:, :, :, edit_one_step.shape[3] // 2:] 
    loss += torch.mean(( right_image - left_image ) ** 2)
    loss=loss*100000
    return loss,masked_grad



def zero_low_frequencies(x, ratio):
    
    ratio_log =1- 2 ** ratio
    
    x = x.squeeze(0)
    assert x.size(1) == x.size(2)
    image_size = x.size(1)
    
    x_coords = torch.linspace(0, 1, steps=image_size).view(1, -1).cuda()
    y_coords = torch.linspace(0, 1, steps=image_size).view(1, -1).cuda()
    
    x_mask = torch.sigmoid((x_coords - ratio_log[0]) * 100 )
    y_mask = torch.sigmoid((y_coords - ratio_log[1]) * 100 )
    x_mask = torch.flip(x_mask, dims=[1])
    y_mask = torch.flip(y_mask, dims=[1])
    
    
    mask = torch.matmul(x_mask.T, y_mask) 
    
    x_dct = dct.dct_2d(x) 
    
    x_dct *= mask
    
    x_idct = dct.idct_2d(x_dct) 
    
    x = x_idct.unsqueeze(0)

    return x

# for image gen
def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0




## for eval
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

def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image

def prepare_image(image):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image[0]

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



def lpips_(a, b ):
    import lpips

    lpips_score = lpips.LPIPS(net='alex').to(a.device)
    return lpips_score(a, b)


def image_align(a, b):
    
    pass

def vifp_(p1,p2):
    i1 = cv2.imread(p1)
    i2 = cv2.imread(p2)
    return vifp(i1,i2)

def ssim_(p1, p2):
    i1 = cv2.imread(p1)
    i2 = cv2.imread(p2)
    i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
    
    return ssim(i1, i2)

from math import log10, sqrt

def psnr_(a, b):

    original = cv2.imread(a)
    compressed = cv2.imread(b)
    
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def filter_list(file_paths, source_list):

    indices = sorted([int(os.path.basename(path).split('_')[0]) for path in file_paths])
    return [source_list[i] for i in indices]



def clip_direction_(a,b):
    e_image_edit = a.image_embeds
    e_image_source = b.image_embeds
    e_text_edit = a.text_embeds
    e_text_source = b.text_embeds

    image_direction = e_image_edit - e_image_source  
    text_direction = e_text_edit - e_text_source  
    image_direction_norm = image_direction / image_direction.norm(dim=1, keepdim=True)
    text_direction_norm = text_direction / text_direction.norm(dim=1, keepdim=True)

    directional_similarity = torch.sum(image_direction_norm * text_direction_norm, dim=1)

    return directional_similarity
def clip_(a,b,c=None):
    import json

    source_caption_path = "../../ImageNet-Edit_original_caption.json"
    with open(source_caption_path , 'r') as source_path:
        source_caption = json.load(source_path)
    source_list = list(source_caption.values())
    filtered_source_list=filter_list(a, source_list)
    
    gen_prompt_path = "../../ImageNet-Edit_prompt.json"
    with open(gen_prompt_path , 'r') as prompt_path:
        prompt = json.load(prompt_path)

    prompt_list = list(prompt.values())
    filtered_prompt_list=filter_list(a, prompt_list)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    
    adv_images = [Image.open(img_path) for img_path in a]
    clean_images = [Image.open(img_path) for img_path in b]

    inputs_adv = processor(text=filtered_prompt_list, images=adv_images, return_tensors="pt", padding=True)
    inputs_clean = processor(text=filtered_source_list , images=clean_images, return_tensors="pt", padding=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs_adv = inputs_adv.to(device)
    inputs_clean = inputs_clean.to(device)
    outputs_adv = model(**inputs_adv)
    outputs_clean = model(**inputs_clean)

    similarities_ia = torch.cosine_similarity(outputs_adv.image_embeds.unsqueeze(1), outputs_clean.image_embeds.unsqueeze(0), dim=2)
    clip_direction= None


    if c is not None:
        source_images = [Image.open(img_path) for img_path in c]

        inputs_source = processor(text=filtered_source_list, images=source_images, return_tensors="pt", padding=True)

        inputs_source = inputs_source.to(device)
        outputs_source = model(**inputs_source)
        
        clip_direction= clip_direction_(outputs_adv,outputs_source)



    
    return similarities_ia,clip_direction,outputs_clean.logits_per_image/100 ,outputs_adv.logits_per_image/100

def sync_files(a, b):

    a_files = set(f for f in os.listdir(a) if not f.startswith('.ipynb'))  
    
    b_files = set(f for f in os.listdir(b) if not f.startswith('.ipynb'))  


    common_files = a_files.intersection(b_files)

    base_path = os.path.dirname(os.path.normpath(a))  
    new_dir = os.path.join(base_path, 'none')

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    for file_name in common_files:
        src_path = os.path.join(b, file_name)
        dest_path = os.path.join(new_dir, file_name)
        shutil.copy2(src_path, dest_path)
        
        
def fid_(a,b):

    sync_files(a,b)
    base_path = os.path.dirname(os.path.normpath(a))  
    c = os.path.join(base_path, 'none')
    result = subprocess.run(
                ['python', '-m', 'pytorch_fid', c, a],
                capture_output=True, text=True
            )
    return float(result.stdout.strip().split()[-1])

def image_reward_(a,b):
    import json
    scores=[]
    model = reward.load("ImageReward-v1.0")
    if b =='img':
        source_caption_path = "../../ImageNet-Edit_original_caption.json"
        with open(source_caption_path , 'r') as source_path:
            source_caption = json.load(source_path)
        source_list = list(source_caption.values())
        filtered_source_list=filter_list(a, source_list)
        for index in range(len(a)):
            score = model.score(filtered_source_list[index], a[index])
            scores.append(score)
            
    elif b =='gen': 
        gen_prompt_path = "../../ImageNet-Edit_prompt.json"
        with open(gen_prompt_path , 'r') as prompt_path:
            prompt = json.load(prompt_path)
        prompt_list = list(prompt.values())
        filtered_prompt_list=filter_list(a, prompt_list)

        for index in range(len(a)):
                score = model.score(filtered_prompt_list[index], a[index])
                scores.append(score)

    return scores
    
def psnr():
    pass
