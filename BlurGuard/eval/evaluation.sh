#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# cd to eval directory
cd "$(dirname "$0")"
## source path for eval.py
source="./source"
# image paths to evaluate : 'img' folder's parent folder
paths=(
    ../out/BlurGuard
)

# evaluate each image path
for path in "${paths[@]}"; do
    echo "======================================"
    echo "[INFO] current path: $path"
    echo "======================================"
    
    # 1) eval.py -> calculate metrics
    python eval_img.py --path "$path" --source "$source"

    #2) result.py -> calculate metrics to csv "out_stat"
    result_path="$path/out_stat"
    python result_img.py --path "$result_path"
done
