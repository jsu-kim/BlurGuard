#!/usr/bin/env bash

# choose subdirectories to generate images (purified images)
subdirs=("img" "jpeg" "jpeg_ups" "noise_ups" "grid_pure" "diff_pure" "pdm_pure" "impress") # you can add or remove subdirectories as you want
prompt_path="../../ImageNet-Edit_prompt.json"
protected_img_dir="../out/BlurGuard"

for sub in "${subdirs[@]}"; do
    adv_image_path="${protected_img_dir}/${sub}/"
    gen_image_path="${protected_img_dir}/gen/${sub}/"

    python generate_image.py \
        --adv_image_path "$adv_image_path" \
        --gen_image_path "$gen_image_path" \
        --prompt_path "$prompt_path"
done