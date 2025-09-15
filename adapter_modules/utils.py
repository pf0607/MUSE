import os
import random

import numpy as np
from PIL import ImageDraw, ImageFont
import torch

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def save_generated_images(images, boxes, phrases, save_path, save_img_name, prompt, width, height):
    os.makedirs(save_path, exist_ok=True)
    font_path = 'adapter_modules/DejaVuSerif.ttf'
    font = ImageFont.truetype(font_path , 30)
    for i, image in enumerate(images):
        image.save(os.path.join(save_path, f"{save_img_name}_{i}.jpg"))
        draw = ImageDraw.Draw(image)
        for obj_i, bbox in enumerate(boxes[0]):
            color = np.concatenate([255.0 * np.random.random(3),], axis=0).astype(np.uint8)
            color = tuple(color)
            x1,y1,x2,y2 = bbox
            x1,x2 = int(x1*width),int(x2*width)
            y1,y2 = int(y1*height),int(y2*height)
            draw.rectangle([x1,y1,x2,y2], outline= color )
            draw.text( (x1,y1), phrases[0][obj_i] , fill = color, font= font)
        name = save_img_name if save_img_name is not None else prompt
        image.save(os.path.join(save_path, f"anno_{name}_{i}.jpg"))

