import os
import torch
from diffusers import StableDiffusionXLPipeline,StableDiffusionPipeline
# from adapter_modules.pipeline_stable_diffusion_xl import StableDiffusionXLCustomPipeline as StableDiffusionXLPipeline
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPImageProcessor

from adapter_modules.resampler import Resampler
from adapter_modules.model import MUSE_Adapter
import random

import sys
import numpy as np
import cv2
import time
import itertools
from adapter_modules.adapter import PositionNetPLUS as PositionNet
import time
from adapter_modules.utils import make_a_sentence
import json

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


class MUSE_pipeline(torch.nn.Module):
    def __init__(self, muse_ckpt):
        super().__init__()
        #https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
        base_model_path = "./SDXL"
        # https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
        image_encoder_path = "./CLIP"
        
        self.device = "cuda"
        self.image_processor = CLIPImageProcessor()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(self.device, dtype=torch.float16)

        # load SDXL pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            add_watermarker=False,
        )
        self.pipe.to(self.device)
        
        max_box_num = 10
        num_tokens_image = 4
        num_tokens_text = 1

        self.image_proj_model = PositionNet(in_dim_text=768, in_dim_image=self.image_encoder.config.hidden_size,
                out_dim_text=2048, out_dim_image=self.pipe.unet.config.cross_attention_dim, fourier_freqs=16, num_tokens_text=num_tokens_text, num_tokens_image = num_tokens_image
                ).to(self.device, dtype=torch.float16)

        self.muse_model = MUSE_Adapter(self.pipe.unet, self.image_proj_model, ckpt_path=muse_ckpt, device=self.device, num_tokens=num_tokens)

    def generate(
        self,
        imgs=["./examples/example_dog.jpg", "./examples/example_cat.jpg"], 
        phrases = [["dog", "cat"]], 
        boxes = [[[0., 0.25, 0.4, 0.75], [0.6, 0.25, 1., 0.75]]], 
        prompt = "best quality, high quality, a dog and cat on the beach", 
        sample_name = None,
        num_samples=4, 
        result_path = "./res",
        log_id = 'test',
        height = 512,
        width = 512,
        item_idx = 0,
        max_box_num = None,
        use_mask =False,
        scale =1.0
        ):
        
        if use_mask:
            input_images = [get_mask_image(img) for img in imgs]
        else:
            input_images = [Image.open(img) for img in imgs]
        input_images = [x.convert("RGB").resize((512, 512)) for x in input_images]

        for i in range(max_box_num - len(phrases[0])):
            phrases[0].append("")

        masks= torch.zeros([1, max_box_num]).to(self.device, dtype = torch.float16)
        masks[0, :len(imgs)] =1

        text_masks = masks
        image_masks = masks

        images = self.muse_model.generate(pipe=self.pipe, num_samples=num_samples, num_inference_steps=30, seed=0,
                                prompt=[prompt], scale=scale, boxes=boxes, phrases=phrases, height=height, width=width ,
                                max_box_num = max_box_num, masks=masks, text_masks=text_masks, 
                                pil_images=input_images, image_masks=image_masks, image_encoder=self.image_encoder,
                                image_processor=self.image_processor
                                )

        font_path = 'DejaVuSerif.ttf'
        font = ImageFont.truetype(font_path , 30)
        
        if result_path and log_id:
            save_path = os.path.join(result_path, log_id)
            os.makedirs(save_path, exist_ok=True)

            for i, image in enumerate(images):
                image.save(os.path.join(save_path, f"{sample_name}_{i}.jpg"))
                draw = ImageDraw.Draw(image)
                for obj_i, bbox in enumerate(boxes[0]):
                    color = np.concatenate([255.0 * np.random.random(3),], axis=0).astype(np.uint8)
                    color = tuple(color)
                    x1,y1,x2,y2 = bbox
                    x1,x2 = int(x1*width),int(x2*width)
                    y1,y2 = int(y1*height),int(y2*height)
                    draw.rectangle([x1,y1,x2,y2], outline= color )
                    draw.text( (x1,y1), phrases[0][obj_i] , fill = color, font= font) 
                if sample_name is None:
                    sample_name = prompt
                image.save(os.path.join(save_path, f"anno_{sample_name}_{i}.jpg"))
        
        return images

if __name__ == "__main__":
    muse_ckpt = "muse_weight.pth"
    num_tokens = 4
    log_id = f'inference3'
    result_path="result"
    pipeline = MUSE_pipeline(muse_ckpt)

    seed_everything(0)

    imgs = ["imgs/cat1.jpg", "imgs/dog1.jpg", "imgs/dog2.jpg"]
    phrases = [["cat", "dog", "dog"]]
    boxes = [[[0.1, 0.1, 0.55, 0.6], [0.45, 0.3, 0.75, 0.6], [0.7, 0.4, 0.9, 0.6]]]
    prompt = "a photo of a cat , a dog and a dog."
    sample_name = prompt
    
    # height = 512
    # width = 512
    # images = pipeline.generate(
    #     imgs = imgs,
    #     phrases = phrases ,   
    #     boxes = boxes, 
    #     prompt = prompt,
    #     num_samples=1,      
    #     log_id = log_id,
    #     result_path = result_path,
    #     height = height,
    #     width = width,
    #     max_box_num =10,
    #     sample_name = sample_name,
    #     use_mask=False,
    #     scale = 0.8,
    #     )


    sample_name = prompt + "_1024"
    height = 1024
    width = 1024
    images = pipeline.generate(
        imgs = imgs,
        phrases = phrases ,   
        boxes = boxes, 
        prompt = prompt,
        num_samples=1,      
        log_id = log_id,
        result_path = result_path,
        height = height,
        width = width,
        max_box_num =10,
        sample_name = sample_name,
        use_mask=False,
        scale = 0.8,
        )
