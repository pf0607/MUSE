import os
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .resampler import Resampler
from torch.cuda.amp import autocast

import torch.nn as nn



class FourierEmbedder():
    def __init__(self, num_freqs=64, temperature=100):

        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** ( torch.arange(num_freqs) / num_freqs )  

    @ torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        "x: arbitrary shape of tensor. dim: cat dim"
        out = []
        for freq in self.freq_bands:
            out.append( torch.sin( freq*x ) )
            out.append( torch.cos( freq*x ) )
        return torch.cat(out, cat_dim)



class Image_Proj(torch.nn.Module):
    def __init__(self, in_dim_image, position_dim, mid_dim, out_dim, num_tokens_image):
        super().__init__()
        self.in_dim_image = in_dim_image
        self.out_dim = out_dim 
        self.position_dim = position_dim
        self.num_tokens_image = num_tokens_image

        self.resampler = Resampler(
            dim=out_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=num_tokens_image,
            embedding_dim=in_dim_image,
            output_dim=out_dim,
            ff_mult=4
        )

        self.linears_bbox = nn.Sequential(
            nn.Linear( self.position_dim, mid_dim),
            nn.SiLU(),
            nn.Linear( mid_dim, mid_dim),
            nn.SiLU(),
            nn.Linear( mid_dim, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, out_dim ),
        )
    

    def forward(self, image_embeddings, xyxy_embeddings, B, N):
        image_embeddings = self.resampler(image_embeddings)
        image_embeddings = image_embeddings.view(B, N, image_embeddings.shape[-2], image_embeddings.shape[-1])  # (bsz, rn, num_tokens, cross_attention_dim)

        xyxy_embeddings_image = self.linears_bbox( xyxy_embeddings ).view(B, N, self.out_dim).unsqueeze(2).repeat(1, 1, self.num_tokens_image, 1)
        objs_image = (image_embeddings + xyxy_embeddings_image).view(B*N,self.num_tokens_image,-1).view(B, -1, self.out_dim)

        return objs_image


class PositionNetPLUS(torch.nn.Module):
    def __init__(self, in_dim_text, in_dim_image, out_dim_text, out_dim_image, fourier_freqs=8, 
                num_tokens_image=4, num_tokens_text=1):
        super().__init__()
        self.in_dim_image = in_dim_image
        self.in_dim_text = in_dim_text
        self.out_dim_text = out_dim_text 
        self.out_dim_image = out_dim_image

        self.num_tokens_image = num_tokens_image
        self.num_tokens_text = num_tokens_text     

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*4 # 2 is sin&cos, 4 is xyxy 

        mid_dim = 3072

        self.linears_text = nn.Sequential(
            nn.Linear( self.in_dim_text + self.position_dim, mid_dim),
            nn.SiLU(),
            nn.Linear( mid_dim, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, out_dim_text ),
        )

        self.image_proj = Image_Proj(in_dim_image, self.position_dim, mid_dim, out_dim_image, num_tokens_image)

    def forward(self, boxes, text_embeddings, image_embeddings):

        B, N, _ = boxes.shape 

        xyxy_embeddings_text = self.fourier_embedder(boxes) # B*N*4 --> B*N*C

        objs_text  = self.linears_text(torch.cat([text_embeddings, xyxy_embeddings_text], dim=-1)).view(B, N, self.out_dim_text)

        objs_image = self.image_proj(image_embeddings, xyxy_embeddings_text, B, N)
        # assert objs.shape == torch.Size([B, N*(self.num_tokens_image+self.num_tokens_text), self.out_dim])        
        return objs_text, objs_image






