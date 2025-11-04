
import os
import argparse
from skimage.io import imread, imsave
from skimage import img_as_ubyte, img_as_float32
from PIL import Image
import numpy as np
import torch

def add_gaussian_noise(image, stddev, seed=42):
    np.random.seed(seed)  
    noise = np.random.normal(0, stddev, image.shape).astype(np.float32)

    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image

def add_noise_to_image(image, noise_strength, seed=42):
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))  


    img_float = img_as_float32(image)


    noisy_img = add_gaussian_noise(img_float, noise_strength, seed=seed)


    noisy_img = img_as_ubyte(noisy_img)


    noisy_img_pil = Image.fromarray(noisy_img)
    return noisy_img_pil

def process_image(input_file, output_file, noise_strength, seed=42):

    image = Image.open(input_file)
    noisy_image = add_noise_to_image(image, noise_strength, seed=seed)
    noisy_image.save(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--adv_image_path', type=str, required=True, help="Input image folder")
    parser.add_argument('--out_image_path', type=str, required=True, help="Output image folder")
    parser.add_argument('--noise_strength', type=float, default=0.1, help="Noise strength (default: 0.1)")

    args = parser.parse_args()


    seed = 42  
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    image_folder = args.adv_image_path
    output_folder = args.out_image_path
    noise_strength = args.noise_strength

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    for image_name in os.listdir(image_folder):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            input_file = os.path.join(image_folder, image_name)
            output_file = os.path.join(output_folder, image_name)
            process_image(input_file, output_file, noise_strength, seed=seed)
            print(f"Processed {image_name}")

    print("Processing complete.")


