# MUSE

Official codes of [MUSE: Multi-Subject Unified Synthesis via Explicit Layout Semantic Expansion ](https://arxiv.org/abs/2508.14440) [ICCV2025].
<img width="1838" height="748" alt="image" src="https://github.com/user-attachments/assets/968e6a31-4ed5-471a-b910-7f52e851d1ae" />

## Requirement

```
conda create -n muse python=3.8
conda activate muse
pip install -r requirements.txt
```

## Model Preparation



1. **Download Base Models**: Download the pretrained [SDXL-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [CLIP-G](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) models.
2. **Download MUSE Checkpoint**: Download our [MUSE](https://huggingface.co/pf0607/MUSE) model checkpoint.

## Inference

We provide demos for synthesizing images with **2** and **3** subjects, supporting both **512x512** and **1024x1024** resolutions.

Before running, make sure to replace the placeholder paths with the actual paths to your model files.

**2 Subjects Inference**:

```
python inference.py
```

**3 Subjects Inference**:

```
python inference3.py
```

## Citation

If you find this work helpful, please consider citing:

```
@article{peng2025muse,
  title={MUSE: Multi-Subject Unified Synthesis via Explicit Layout Semantic Expansion},
  author={Peng, Fei and Wu, Junqiang and Li, Yan and Gao, Tingting and Zhang, Di and Fu, Huiyuan},
  journal={arXiv preprint arXiv:2508.14440},
  year={2025}
}
```
