#!/usr/bin/env bash
# directory of the protected images as input
protected_img_dir="../out/BlurGuard"


#GriDPure
python gridpure.py \
    --input_dir="$protected_img_dir/img" \
    --output_dir="$protected_img_dir/grid_pure" \
    --pure_steps=10 \
    --pure_iter_num=20 \
    --gamma=0.1

# DiffPure
python diffpure.py \
    --input_dir "$protected_img_dir/img" \
    --output_dir "$protected_img_dir/diff_pure" \
    --pure_steps 100


# Impress
python pg_mask_pur_helen.py --adv_dir="${protected_img_dir}/img" 

# JPEG Compression
python jpeg.py \
    --adv_image_path "$protected_img_dir/img" \
    --out_image_path "$protected_img_dir/jpeg"

# Gaussian Noise
python noise.py \
    --adv_image_path "$protected_img_dir/img" \
    --out_image_path "$protected_img_dir/noise"

# JPEG Upscaling
python up.py \
    --adv_image_path "$protected_img_dir/jpeg" \
    --out_image_path "$protected_img_dir/jpeg_ups"

# Noisy Upscaling
python up.py \
    --adv_image_path "$protected_img_dir/noise" \
    --out_image_path "$protected_img_dir/noise_ups"

# PDM-Pure
eval "$(conda shell.bash hook)"
conda activate deepfloyd # for PDM-Pure, you need to create another conda environment, following the instructions in https://github.com/deep-floyd/IF (Source code is already in this repository : ./PDM-Pure)

cd ./PDM-Pure
python pdm_pure.py --input_dir "$protected_img_dir" --output_dir "$protected_img_dir/pdm_pure"
cd ..