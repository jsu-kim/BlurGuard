import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision
import lpips
from torch.optim import Adam, SGD
import torch_dct as dct
from tqdm import tqdm
from ldm.models.diffusion.ddim import DDIMSampler
import torchvision.transforms as transformsc
import os
from transformers import CLIPTokenizer, CLIPTextModel

ATTACK_TYPE = ['pgd_freq']
import torch

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fft_fps(img_tensor):
    if img_tensor.ndim == 4:  
        img_tensor = img_tensor[0]  
    if img_tensor.shape[0] != 3:  
        img_tensor = img_tensor.permute(2, 0, 1) 

    image = img_tensor / 2 + 0.5
    img_f = torch.fft.fft2(image)
    fps_per_channel = torch.abs(img_f) ** 2
    fps = fps_per_channel.sum(dim=0)
    return fps

def compute_histogram_fft(magnitude_spectrum):

    height, width = magnitude_spectrum.shape
    device = magnitude_spectrum.device


    center_y, center_x = (height - 1) / 2, (width - 1) / 2


    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    r = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    r_int = r.round().long()
    r_max = r_int.max().item() + 1 
    
    masks = torch.stack([(r_int == i).float() for i in range(r_max)], dim=0)  
    freq_mag = (masks * magnitude_spectrum).sum(dim=(-1, -2)) 
    return freq_mag


def filter_delta(log_sigmas, x, mask_dict):
    n_masks = len(mask_dict)
    parts = []
    for i in range(n_masks):
        mask = mask_dict[f'mask{i + 1}'].to(x.device)
        part_pert = gaussian_blur(x, log_sigmas[i].exp()) * mask
        parts.append(part_pert)
    pert_agg = sum(parts)

    return pert_agg


def gaussian_blur(x, sigma, width=33):
    """Applies gaussian blur"""
    is_singleton = (x.ndim == 3)
    if is_singleton:
        x = x[None]
    
    n, channels, h, w = x.shape
    width = width + (width + 1) % 2 

    distance = torch.arange(
        -(width // 2), width // 2 + 1, dtype=torch.float, device=x.device
    )
    gaussian = torch.exp(
        -(distance[:, None] ** 2 + distance[None] ** 2) / (2 * sigma ** 2)
    )
    gaussian = gaussian / gaussian.sum()
    channels = x.size(1)
    kernel = gaussian[None, None].expand(channels, -1, -1, -1)
    
    x = torch.nn.ReflectionPad2d(width // 2)(x)
    y = torch.nn.functional.conv2d(x, kernel, groups=channels)
    if is_singleton:
        y = y[0]
        
    return y

 
    
class LF_PGD():
    def __init__(self, net, epsilon, steps, targeted=True, masks_dict=None, filename="", eps_sigma=0.01,  sigma_weighting=10000 ,learning_rate=0.001,attack_type='pgd_freq'):
        self.net = net
        self.eps = epsilon 
        self.iters = steps
        self.masks_dict = masks_dict
        self.attack_type = attack_type
        self.filename = filename
        self.eps_sigma=eps_sigma
        self.sigma_weighting=sigma_weighting
        self.learning_rate=learning_rate
        self.step_size=1


        if self.attack_type not in ATTACK_TYPE:
            raise AttributeError(f"self.attack_type should be in {ATTACK_TYPE}, \
                                 {self.attack_type} is an undefined")

    def perturb(self, X, y):
        if self.attack_type == 'pgd_freq':
            return self.pgd_freq(X, y)


        else:
            raise NotImplementedError(f"Unknown attack_type: {self.attack_type}")
            
    def pgd_freq(self, X, y):
        """
        - sigma: updated by optimizer (log_sigmas)
        - pert: updated manually with gradient from total_loss
        """
        device = X.device
        X = X.to(device)

        pert = torch.zeros_like(X, requires_grad=True).to(device)

        num_masks = len(self.masks_dict) if self.masks_dict else 0
        log_sigmas = torch.full((num_masks,), 0.0, device=device, requires_grad=True)

        optimizer = Adam([log_sigmas], lr=self.learning_rate)
        pbar = tqdm(range(self.iters))

        adv_loss_list = []
        perceptual_loss_list = []
        sigma_loss_list = []
        all_sigmas = [[] for _ in range(num_masks)]
        
        for i in pbar:
            actual_step_size = self.step_size - (self.step_size - self.step_size / 100) / self.iters * i
            

            optimizer.zero_grad()
            pert.requires_grad_(True)  


            pert_lf = filter_delta(log_sigmas.detach(),pert, self.masks_dict)
 
            pert_lf = torch.clamp(pert_lf, -self.eps, self.eps) 
            X_adv = (X + pert_lf).clamp(-1.0, 1.0)

            embedding = self.net.model.get_first_stage_encoding(self.net.model.encode_first_stage(X_adv))
            adv_loss = (embedding ** 2).mean()

            freq_src = fft_fps(X)
            random_pert=torch.randn_like(X).cuda()
            random_pert=random_pert.clamp(-self.eps,self.eps)
            pert_rob_lf = filter_delta(log_sigmas, random_pert, self.masks_dict)
            pert_rob_lf = torch.clamp(pert_rob_lf, -self.eps, self.eps)
            X_rob = (X + pert_rob_lf).clamp(-1.0, 1.0)
            freq_init =fft_fps(X_rob)
            freq_adv = freq_init.clone().requires_grad_(True)
            fps_src = compute_histogram_fft(freq_src)
            fps_adv = compute_histogram_fft(freq_adv)
            
            lfs = torch.log10(fps_src+1e-8) 
            lfa = torch.log10(fps_adv+1e-8) 


            freq_diff = torch.abs(lfs-lfa)
            sigma_loss = torch.relu(torch.max(freq_diff) - self.eps_sigma) # l inf

            if i < 50:
                total_loss = self.sigma_weighting * sigma_loss
            else:
                total_loss = adv_loss +self.sigma_weighting * sigma_loss


            total_loss.backward()

            grad_pert = pert.grad
            if i < 50:
                optimizer.step()

            if grad_pert != None:
                with torch.no_grad():

                    grad_norms = torch.norm(grad_pert.reshape(X.size(0), -1), p=2, dim=1) + 1e-10
                    grad_pert = grad_pert / grad_norms.view(X.size(0), 1, 1, 1) * 2
                    pert -= actual_step_size * grad_pert 
                    pert = pert.detach_().requires_grad_()

            adv_loss_list.append(float(adv_loss.item()))
            perceptual_loss_list.append(-1)
            sigma_loss_list.append(float(sigma_loss.item()))
        
            log_sigmas_exp = log_sigmas.exp().detach().cpu().numpy().round(4).tolist()
            log_sigmas_exp = [f"{sigma:.4f}" for sigma in log_sigmas_exp] 
            log_sigmas_exp = log_sigmas.exp().detach().cpu().numpy().round(4).tolist()
            for j, sigma in enumerate(log_sigmas_exp):
                all_sigmas[j].append(sigma)
            pbar.set_description(
                f"[pgd_freq] adv:{adv_loss.item():.4f}, sigma_loss:{sigma_loss.item():.4f}, sigmas={log_sigmas_exp}"
            )


        pert_lf = filter_delta(log_sigmas, pert, self.masks_dict)
        pert_lf = torch.clamp(pert_lf, -self.eps, self.eps)
        
        X_adv = (X + pert_lf).clamp(-1.0, 1.0).detach()
        final_sigmas = log_sigmas.exp().detach().cpu().numpy().tolist()

        return X_adv, adv_loss_list, perceptual_loss_list, sigma_loss_list, all_sigmas
                
    
    @staticmethod
    def mp(p):
        import os
        first_dot = p.find('.')
        last_slash = p.rfind('/')
        if first_dot < last_slash:
            assert ValueError('Input path seems like a/b.c/d/g, which is not allowed :(')
        p_new = p[:last_slash] + '/'
        if not os.path.exists(p_new):
            os.makedirs(p_new)


