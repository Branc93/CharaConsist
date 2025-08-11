<div align="center">

## CharaConsist: Fine-Grained Consistent Character Generation

Official implementation of ICCV 2025 paper - CharaConsist: Fine-Grained Consistent Character Generation

[[Paper](https://arxiv.org/abs/2507.11533)] &emsp; [[Project Page](https://murray-wang.github.io/CharaConsist/)] &emsp; <br>
</div>

## Quickstart (Gradio UI)

This repository includes a Gradio app (`app.py`) that wraps the three notebooks (`gen-bg_fg`, `gen-fg_only`, `gen-mix`) into tabs and runs on top of the Diffusers FLUX.1 pipeline with optimizations (fp16, device_map="auto", CPU offload, VAE slicing; xFormers optional).

### Installation

We recommend creating a virtual environment, installing a CUDA-enabled PyTorch matching your driver, then the remaining requirements.

```bash
# (optional) create & activate venv
python -m venv .venv
# On Windows
.venv\Scripts\activate

# Install a CUDA build of PyTorch (example for CUDA 12.1 on Windows)
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install project deps
pip install -r requirements.txt
```

Notes:
- If you prefer CPU-only for testing, install PyTorch from the CPU index-url instead.
- xFormers is optional; the app runs without it. If you later install a matching wheel, the app will enable it automatically.

### Usage

```bash
python app.py
```

In the UI:
- Leave "Model path" empty and keep "Auto-download" checked to fetch `black-forest-labs/FLUX.1-dev` into `./flux_model` automatically.
- If the model is gated, paste your Hugging Face token or set `HF_TOKEN`/`HUGGINGFACE_TOKEN` in your environment.
- Choose the tab (Background+Foreground, Foreground Only, or Mix), fill prompts, and Generate.

Tips:
- Set `share=True` in `demo.launch()` inside `app.py` to get a public link.
- For Windows, if you see symlink warnings, you can ignore them or enable Developer Mode to improve caching.

## Update
- [x] We provide an independent implementation of the training-free mask extraction and point matching strategies in [point_and_mask](https://github.com/Murray-Wang/CharaConsist/tree/main/point_and_mask), for those who are interested in these details and need to use them separately.
---

## Key Features
- Without any training, CharaConsist can effectively enhance the consistency of text-to-image generation results.
- While maintaining foreground character consistency, CharaConsist can also optionally preserve background consistency, thereby meeting the needs of different application scenarios.
- Built upon the DiT model (FLUX.1), CharaConsist effectively leverages the advantages of the pre-trained model and achieves superior generation quality compared to previous approaches.
- The implementation of CharaConsist includes training-free mask extraction and point matching strategies, which can serve as effective tools for related tasks such as image editing.

## Qualitative Results
<div align="center">Fig. 1 Consistent Character Generation in a Fixed Background.</div>

<a name="fig1"></a>
![Consistent Character Generation in a Fixed Background.](docs/static/images/fg_bg-all.jpg)

<div align="center">Fig. 2 Consistent Character Generation across Different Backgrounds.</div>

<a name="fig2"></a>
![Consistent Character Generation across Different Backgrounds.](docs/static/images/fg_only-all.jpg)

<div align="center">Fig. 3 Story Generation.</div>

<a name="fig3"></a>
![Story Generation.](docs/static/images/story.jpg)

## How to use

### Quick Start
We provide two ways to use CharaConsist for generating consistent characters:

#### (1) Notebook for Single Example
We provide three Jupyter notebooks: 
- `gen-bg_fg.ipynb`: generating consistent character in a fixed background, as shown in [Fig.1](#fig1).
- `gen-fg_only.ipynb`: generating consistent character across different backgrounds, as shown in [Fig.2](#fig2).
- `gen-mix.ipynb`: generating the same character in partly fixed and partly varying backgrounds, as shown in [Fig.3](#fig3).

Users can refer to the detailed descriptions in the notebooks to familiarize themselves with the entire framework of the method.


#### (2) Script for Batch Generation
We provide a batch generation script in `inference.py`. Its functionality is essentially the same as the notebooks above, but it is more convenient for multiple samples generation. Its input parameters include:

- `init_mode`: Different model initialization methods depending on available GPU memory and number of GPUs.
    | init_mode   | initialization | GPU memory   | GPU number  |
    |--------|------|------------|------|
    | 0   | single GPU   |   37 GB   | 1 |
    | 1   | single GPU, with model cpu offload   |  26 GB  | 1 |
    | 2   | multiple GPUs, memory distribute evenly    |  <= 20 GB | >=2 |
    | 3   | single GPU, with sequential cpu offload   | 3 GB | 1 |

- `gpu_ids`: The ids of the GPUs to use. When init_mode is set to 0, 1, or 3, only the first GPU in this id list will be used. When init_mode is set to 2, the memory usage will be evenly distributed across all specified GPUs.
- `prompts_file`: Path to the file containing the input prompts. Two examples are provided in the `examples` folder.
- `model_path`: Path to the pre-trained [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) model weights.
- `out_dir`: The path where the output results will be saved.
- `use_interpolate`: Whether to use adaptive token merge. Enabling it improves consistency but increases CPU memory consumption.
- `share_bg`: Whether to preserve the background unchanged
- `save_mask`: Whether to save the automatically extracted masks during the generation process for visualization

Generating consistent character in a fixed background:
```bash
python inference.py \
--init_mode 0 \
--prompts_file examples/prompts-bg_fg.txt \
--model_path path/to/FLUX.1-dev \
--out_dir results/bg_fg \
--use_interpolate --save_mask --share_bg
```

Generating consistent character across different backgrounds:
```bash
python inference.py \
--init_mode 0 \
--prompts_file examples/prompts-fg_only.txt \
--model_path path/to/FLUX.1-dev \
--out_dir results/fg_only \
--use_interpolate --save_mask
```

## BibTeX
If you find CharaConsist useful for your research and applications, please cite using this BibTeX:

```BibTeX
@inproceedings{CharaConsist,
  title={{CharaConsist}: Fine-Grained Consistent Character Generation},
  author={Wang, Mengyu and Ding, Henghui and Peng, Jianing and Zhao, Yao and Chen, Yunpeng and Wei, Yunchao},
  booktitle={ICCV},
  year={2025}
}
```
