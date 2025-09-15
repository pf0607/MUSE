import os
from typing import List

from diffusers.pipelines.controlnet import MultiControlNetModel
import torch

from adapter_modules.attention_processor import AttnProcessor
from adapter_modules.attention_processor import MUSE_AttnProcessor

class MuseAdapter(torch.nn.Module):
    def __init__(self, unet, image_proj_model, adapter_modules=None, ckpt_path=None,
                num_tokens=4, text_tokens=77, device="cuda", controlnet=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        self.num_tokens = num_tokens
        self.text_tokens = text_tokens
        self.device = device
        self.controlnet = controlnet
        self.cross_attention_dim = self.unet.config.cross_attention_dim

        if self.adapter_modules is None:
            self.set_muse_adapter()

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def load_from_checkpoint(self, ckpt_path: str):
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path)

        state_dict = torch.load(ckpt_path, map_location="cpu")

        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        self.image_proj_model.load_state_dict(state_dict["image_proj_model"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["adapter_modules"], strict=True)

        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")



    def set_muse_adapter(self, weight_dtype=torch.float16, cache_attention_maps=True):
        # set attention processor
        attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
                num_heads = self.unet.config.attention_head_dim[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                num_heads = list(reversed(self.unet.config.attention_head_dim))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
                num_heads = self.unet.config.attention_head_dim[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = MUSE_AttnProcessor(
                                                hidden_size = hidden_size,
                                                text_in_dim=2048,   #cross_attention_dim
                                                image_in_dim=2048, 
                                                scale=1.0, 
                                                num_object=10,
                                                num_tokens_text=1, #1
                                                num_tokens_image=4, #4
                                                stage=2).to(self.device, dtype=weight_dtype)

        self.unet.set_attn_processor(attn_procs)
        self.adapter_modules = torch.nn.ModuleList(self.unet.attn_processors.values())
        if self.controlnet is not None:
            if isinstance(self.controlnet, MultiControlNetModel):
                for controlnet in self.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(text_tokens=self.text_tokens, num_tokens=self.num_tokens))
            else:
                self.controlnet.set_attn_processor(CNAttnProcessor(text_tokens=self.text_tokens, num_tokens=self.num_tokens))

    @torch.inference_mode()
    def get_image_embeds(self, processed_images, image_encoder=None, weight_dtype=torch.float16):

        processed_images = processed_images.view(-1, processed_images.shape[-3], processed_images.shape[-2],
                                                 processed_images.shape[-1])  # (bsz*rn, ...)
                                                 
        image_embeds = image_encoder(processed_images.to(self.device, dtype=weight_dtype),
                                         output_hidden_states=True).hidden_states[-2]  # (bsz*rn, num_tokens, embedding_dim)
    

        return image_embeds  # [bsz*rn, ...]

    def set_scale(self, scale, subject_scales):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, MUSE_AttnProcessor):
                attn_processor.scale = scale
                attn_processor.subject_scales = subject_scales

    def enable_psuedo_attention_mask(self, mask_threshold=0.5, start_step=5):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, MUSE_AttnProcessor):
                attn_processor.mask_threshold = mask_threshold
                attn_processor.start_step = start_step
                attn_processor.use_psuedo_attention_mask = True
                attn_processor.need_text_attention_map = True
                attn_processor.attention_maps = []  # clear attention maps

    def generate(self, pipe, pil_images=None, processed_images=None, prompt=None, negative_prompt=None, scale=1.0,
                num_samples=4, seed=None, guidance_scale=7.5, num_inference_steps=30, image_processor=None,
                image_encoder=None, weight_dtype=torch.float16,
                boxes=None, phrases=None, height=1024, width=1024, subject_scales=None,
                max_box_num = None, masks = None, text_masks = None, image_masks = None,
                **kwargs):

        self.pipe = pipe
        self.set_scale(scale, subject_scales)

        bsz = len(boxes)  
        if processed_images is None:
            processed_images = []
            for pil_image in pil_images:
                processed_image = image_processor(images=pil_image, return_tensors="pt").pixel_values
                processed_images.append(processed_image)
            processed_images = torch.stack(processed_images, dim=0)
            if max_box_num:
                blank_processed_images = torch.zeros([bsz, max_box_num, 3, 224, 224], dtype=processed_images.dtype).to(self.device)
                blank_processed_images[:, :processed_images.shape[0]] = processed_images.squeeze(1)
                processed_images = blank_processed_images

        num_prompts = bsz
        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            # negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            negative_prompt = ""  
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        cross_attention_kwargs = {}
        grounding_kwargs = None
        if boxes is not None:
            boxes = torch.tensor(boxes).to(self.device, weight_dtype)

            if max_box_num:
                blank_boxes = torch.zeros([bsz, max_box_num, 4], dtype=boxes.dtype).to(self.device)
                blank_boxes[:, : boxes.shape[1]] = boxes
                boxes = blank_boxes

        with torch.inference_mode():

            image_embeds = self.get_image_embeds(processed_images, image_encoder, weight_dtype=weight_dtype)

            phrase_input_ids=self.pipe.tokenizer(
                                phrases[0],
                                max_length=self.pipe.tokenizer.model_max_length,
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt"
                            ).input_ids

            encoder_output = self.pipe.text_encoder(phrase_input_ids.view(-1,self.pipe.tokenizer.model_max_length).to(self.device), output_hidden_states=True)  
            text_prompt_embeds_ = encoder_output.pooler_output.view(bsz, max_box_num, -1) 
            
            text_prompt_embeds, image_prompt_embeds = self.image_proj_model(boxes, text_prompt_embeds_, image_embeds)
            image_prompt_embeds = image_prompt_embeds.view(bsz, -1, image_prompt_embeds.shape[-2], image_prompt_embeds.shape[-1])  # (bsz, rn, num_tokens, cross_attention_dim)
            image_prompt_embeds = image_prompt_embeds.view(bsz, image_prompt_embeds.shape[-3] * image_prompt_embeds.shape[-2],
                                                        image_prompt_embeds.shape[-1])  # (bsz, total_num_tokens*rn, cross_attention_dim)
            text_prompt_embeds = text_prompt_embeds.view(bsz, -1, text_prompt_embeds.shape[-2], text_prompt_embeds.shape[-1])  # (bsz, rn, num_tokens, cross_attention_dim)
            text_prompt_embeds = text_prompt_embeds.view(bsz, text_prompt_embeds.shape[-3] * text_prompt_embeds.shape[-2],
                                                        text_prompt_embeds.shape[-1])
            
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
            bs_embed, seq_len, _ = text_prompt_embeds.shape
            text_prompt_embeds = text_prompt_embeds.repeat(1, num_samples, 1)
            text_prompt_embeds = text_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

            blank_processed_images = torch.zeros([bsz, max_box_num, 3, 224, 224], dtype=processed_images.dtype).to(self.device)           
            uncond_image_prompt_embeds_ = self.get_image_embeds(blank_processed_images, image_encoder, weight_dtype=weight_dtype)

            blank_boxes = torch.zeros([bsz, max_box_num, 4], dtype=boxes.dtype).to(self.device)
            blank_phrase_input_ids =  self.pipe.tokenizer(
                            "",
                            max_length=self.pipe.tokenizer.model_max_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt"
                            ).input_ids
            blank_phrase_input_ids = blank_phrase_input_ids.unsqueeze(0).repeat_interleave(max_box_num, dim=0)
            blank_encoder_output = self.pipe.text_encoder(blank_phrase_input_ids.view(-1,self.pipe.tokenizer.model_max_length).to(self.device), output_hidden_states=True)  
            uncond_text_prompt_embeds_ = blank_encoder_output.pooler_output.view(bsz, max_box_num, -1)  
            uncond_text_prompt_embeds, uncond_image_prompt_embeds = self.image_proj_model(blank_boxes, uncond_text_prompt_embeds_, uncond_image_prompt_embeds_)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bsz, -1, uncond_image_prompt_embeds.shape[-2], uncond_image_prompt_embeds.shape[-1])  # (bsz, rn, num_tokens, cross_attention_dim)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bsz, uncond_image_prompt_embeds.shape[-3] * uncond_image_prompt_embeds.shape[-2],
                                                        uncond_image_prompt_embeds.shape[-1])  # (bsz, total_num_tokens*rn, cross_attention_dim)
            uncond_text_prompt_embeds = uncond_text_prompt_embeds.view(bsz, -1, uncond_text_prompt_embeds.shape[-2], uncond_text_prompt_embeds.shape[-1])  # (bsz, rn, num_tokens, cross_attention_dim)
            uncond_text_prompt_embeds = uncond_text_prompt_embeds.view(bsz, uncond_text_prompt_embeds.shape[-3] * uncond_text_prompt_embeds.shape[-2],
                                                        uncond_text_prompt_embeds.shape[-1])
            

            bs_embed, seq_len, _ = uncond_image_prompt_embeds.shape
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
            bs_embed, seq_len, _ = uncond_text_prompt_embeds.shape
            uncond_text_prompt_embeds = uncond_text_prompt_embeds.repeat(1, num_samples, 1)
            uncond_text_prompt_embeds = uncond_text_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

            prompt_embeds_, negative_prompt_embeds_, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            image_prompt_embeds = torch.cat([uncond_image_prompt_embeds,image_prompt_embeds ],dim=0)
            text_prompt_embeds = torch.cat([uncond_text_prompt_embeds,text_prompt_embeds ],dim=0)
            cross_attention_kwargs['text_box_grounding_embeds'] = text_prompt_embeds
            cross_attention_kwargs['image_box_grounding_embeds'] = image_prompt_embeds

            prompt_embeds = prompt_embeds_
            negative_prompt_embeds = negative_prompt_embeds_


        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            cross_attention_kwargs=cross_attention_kwargs,
            height=height,
            width=width,
            **kwargs,
        ).images

        return images
