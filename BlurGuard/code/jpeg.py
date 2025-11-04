from PIL import Image
import os
import argparse
import glob
# Define the paths/combined_image.png
parser = argparse.ArgumentParser()
parser.add_argument('--adv_image_path', type=str)
parser.add_argument('--out_image_path', type=str)
args = parser.parse_args()

image_folder = args.adv_image_path
jpeg_folder = args.out_image_path
# Create the JPEG folder if it doesn't exist
os.makedirs(jpeg_folder, exist_ok=True)

img_paths = glob.glob(image_folder+'*')
img_paths = [p for p in img_paths if 'attacked' in p] 

             

# Create the JPEG folder if it doesn't exist
os.makedirs(jpeg_folder, exist_ok=True)

# Get a list of all PNG files in the source folder
image_paths = [f for f in os.listdir(image_folder) if f.endswith('attacked.png')]

# Convert each image to JPEG format with the specified quality
for image_path in image_paths:
    # Load the image
    img = Image.open(os.path.join(image_folder, image_path)).convert("RGB")
   
    # Define the path for the JPEG image
    jpeg_path = os.path.join(jpeg_folder, os.path.splitext(image_path)[0] + ".jpeg")
   
    # Save the image as JPEG with quality 75
    img.save(jpeg_path, "JPEG", quality=75)
print("All PNG images have been converted to JPEG format with quality 75.")


