# MUSE
Official codes of [MUSE: Multi-Subject Unified Synthesis via Explicit Layout Semantic Expansion ](https://arxiv.org/abs/2508.14440) [ICCV2025].
<img width="1838" height="748" alt="image" src="https://github.com/user-attachments/assets/968e6a31-4ed5-471a-b910-7f52e851d1ae" />
Existing text-to-image diffusion models have demonstrated remarkable capabilities in generating high-quality images guided by textual prompts. However, achieving multi-subject compositional synthesis with precise spatial control remains a significant challenge. In this work, we address the task of layout-controllable multi-subject synthesis (LMS), which requires both faithful reconstruction of reference subjects and their accurate placement in specified regions within a unified image. While recent advancements have separately improved layout control and subject synthesis, existing approaches struggle to simultaneously satisfy the dual requirements of spatial precision and identity preservation in this composite task. To bridge this gap, we propose MUSE, a unified synthesis framework that employs concatenated cross-attention (CCA) to seamlessly integrate layout specifications with textual guidance through explicit semantic space expansion. The proposed CCA mechanism enables bidirectional modality alignment between spatial constraints and textual descriptions without interference. Furthermore, we design a progressive two-stage training strategy that decomposes the LMS task into learnable sub-objectives for effective optimization. Extensive experiments demonstrate that MUSE achieves zero-shot end-to-end generation with superior spatial accuracy and identity consistency compared to existing solutions, advancing the frontier of controllable image synthesis.

<img width="1927" height="777" alt="image" src="https://github.com/user-attachments/assets/99801c4d-2918-4f0e-b580-919328244d25" />

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
