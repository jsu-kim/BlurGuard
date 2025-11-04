# BlurGuard: A Simple Approach for Robustifying Image Protection Against AI-Powered Editing (NeurIPS 2025)

Code for the paper *"BlurGuard: A Simple Approach for Robustifying Image Protection Against AI-Powered Editing"* (NeurIPS 2025).


## BlurGuard Installation and Execution Guide

### Installation

1. **Set up the Conda environment**

   ```bash
   conda env create -f BlurGuard.yml
   conda activate BlurGuard
   ```

2. **Download and prepare the Stable Diffusion weights**

   ```bash
   wget -c https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
   mkdir -p ckpt
   mv sd-v1-4.ckpt ckpt/model.ckpt
   ```

3. **Clone and install Taming Transformers**

   ```bash
   git clone https://github.com/CompVis/taming-transformers.git
   cd taming-transformers
   pip install -e .
   ```

4. **Download the Segment Anything checkpoint**

   ```bash
   cd ../code/segment_anything
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   cd ..
   ```

5. **Prepare the DeepFloyd IF environment for PDM-Pure**

   PDM-Pure is one of the purification tools we evaluate. It relies on DeepFloyd IF and must be run inside its own Python environment. Follow the instructions in the [DeepFloyd IF repository](https://github.com/deep-floyd/IF) to create that environment before executing PDM-Pure.

### Execution

Set the working directory to `BlurGuard/BlurGuard/code`.

1. **Run BlurGuard (protection stage)**

   ```bash
   python blurguard.py
   ```

   Protected images will be written to `BlurGuard/BlurGuard/out/BlurGuard/img/`.

2. **Run purification tools**

   ```bash
   bash purify.sh
   ```

   The script reads from `BlurGuard/BlurGuard/out/BlurGuard/` by default. Run BlurGuard first, then execute this step to purify the protected outputs. Remember to activate the dedicated DeepFloyd IF environment before launching PDM-Pure.

3. **Generate edited images from purified results**

   ```bash
   bash generate_image.sh
   ```

   The prompts used for editing are stored in `BlurGuard/ImageNet-Edit_prompt.json`.

4. **Evaluate protection performance**

   ```bash
   bash ../eval/evaluation.sh
   ```

   Metrics will be saved to `BlurGuard/BlurGuard/out/BlurGuard/out_stat/results.csv`, which reports naturalness (visual quality) and worst-case protection effectiveness across the seven purification tools.
