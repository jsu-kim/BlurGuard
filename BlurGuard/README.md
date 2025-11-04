# BlurGuard: A Simple Approach for Robustifying Image Protection Against AI-Powered Editing (NeurIPS 2025)
Code for the paper *"BlurGuard: A Simple Approach for Robustifying Image Protection Against AI-Powered Editing"* (NeurIPS 2025).


## BlurGuard Installation and Execution Guide

Installation

1. Set up the Conda Environment

conda env create -f BlurGuard.yml
conda activate BlurGuard

2. Download and Prepare the Stable Diffusion Model

wget -c https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
mkdir -p ckpt
mv sd-v1-4.ckpt ckpt/model.ckpt

3. Clone and Install Taming Transformers

git clone https://github.com/CompVis/taming-transformers.git
cd taming-transformers
pip install -e .

4. Download the Segment Anything Model

cd ../code/segment_anything
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..


Execution


1. Run BlurGuard
  
python blurguard.py  

