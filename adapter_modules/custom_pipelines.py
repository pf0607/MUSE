from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg

from .utils import is_torch2_available

from .attention_processor import MOAttnProcessor_decouple_image_Concat_text_1stage as IPAttnProcessor
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionXLPipeline

        >>> pipe = StableDiffusionXLPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class StableDiffusionXLMOPipeline(StableDiffusionXLPipeline):

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None:
            image_embeds, negative_image_embeds = self.encode_image(ip_adapter_image, device, num_images_per_prompt)
            if self.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])
                image_embeds = image_embeds.to(device)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)





# class StableDiffusionXLMOPipeline(StableDiffusionXLPipeline):
#     def set_scale(self, scale):
#         for attn_processor in self.unet.attn_processors.values():
#             if isinstance(attn_processor, MOAttnProcessor):
#                 attn_processor.scale = scale

#     @torch.no_grad()
#     def __call__(  # noqa: C901
#         self,
#         prompt: Optional[Union[str, List[str]]] = None,
#         prompt_2: Optional[Union[str, List[str]]] = None,
#         height: Optional[int] = None,
#         width: Optional[int] = None,
#         num_inference_steps: int = 50,
#         denoising_end: Optional[float] = None,
#         guidance_scale: float = 5.0,
#         negative_prompt: Optional[Union[str, List[str]]] = None,
#         negative_prompt_2: Optional[Union[str, List[str]]] = None,
#         num_images_per_prompt: Optional[int] = 1,
#         eta: float = 0.0,
#         generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
#         latents: Optional[torch.FloatTensor] = None,
#         prompt_embeds: Optional[torch.FloatTensor] = None,
#         negative_prompt_embeds: Optional[torch.FloatTensor] = None,
#         pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
#         negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
#         output_type: Optional[str] = "pil",
#         return_dict: bool = True,
#         callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
#         callback_steps: int = 1,
#         cross_attention_kwargs: Optional[Dict[str, Any]] = None,
#         guidance_rescale: float = 0.0,
#         original_size: Optional[Tuple[int, int]] = None,
#         crops_coords_top_left: Tuple[int, int] = (0, 0),
#         target_size: Optional[Tuple[int, int]] = None,
#         negative_original_size: Optional[Tuple[int, int]] = None,
#         negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
#         negative_target_size: Optional[Tuple[int, int]] = None,
#         control_guidance_start: float = 0.0,
#         control_guidance_end: float = 1.0,
#         mis: float = 0.3,
#         image_text_box_prompt_embeds: Optional[torch.FloatTensor] = None,
#     ):
#         r"""
#         Function invoked when calling the pipeline for generation.

#         Args:
#             prompt (`str` or `List[str]`, *optional*):
#                 The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
#                 instead.
#             prompt_2 (`str` or `List[str]`, *optional*):
#                 The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
#                 used in both text-encoders
#             height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
#                 The height in pixels of the generated image. This is set to 1024 by default for the best results.
#                 Anything below 512 pixels won't work well for
#                 [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
#                 and checkpoints that are not specifically fine-tuned on low resolutions.
#             width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
#                 The width in pixels of the generated image. This is set to 1024 by default for the best results.
#                 Anything below 512 pixels won't work well for
#                 [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
#                 and checkpoints that are not specifically fine-tuned on low resolutions.
#             num_inference_steps (`int`, *optional*, defaults to 50):
#                 The number of denoising steps. More denoising steps usually lead to a higher quality image at the
#                 expense of slower inference.
#             denoising_end (`float`, *optional*):
#                 When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
#                 completed before it is intentionally prematurely terminated. As a result, the returned sample will
#                 still retain a substantial amount of noise as determined by the discrete timesteps selected by the
#                 scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
#                 "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
#                 Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
#             guidance_scale (`float`, *optional*, defaults to 5.0):
#                 Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
#                 `guidance_scale` is defined as `w` of equation 2. of [Imagen
#                 Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
#                 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
#                 usually at the expense of lower image quality.
#             negative_prompt (`str` or `List[str]`, *optional*):
#                 The prompt or prompts not to guide the image generation. If not defined, one has to pass
#                 `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
#                 less than `1`).
#             negative_prompt_2 (`str` or `List[str]`, *optional*):
#                 The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
#                 `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
#             num_images_per_prompt (`int`, *optional*, defaults to 1):
#                 The number of images to generate per prompt.
#             eta (`float`, *optional*, defaults to 0.0):
#                 Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
#                 [`schedulers.DDIMScheduler`], will be ignored for others.
#             generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
#                 One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
#                 to make generation deterministic.
#             latents (`torch.FloatTensor`, *optional*):
#                 Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
#                 generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
#                 tensor will ge generated by sampling using the supplied random `generator`.
#             prompt_embeds (`torch.FloatTensor`, *optional*):
#                 Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
#                 provided, text embeddings will be generated from `prompt` input argument.
#             negative_prompt_embeds (`torch.FloatTensor`, *optional*):
#                 Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
#                 weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
#                 argument.
#             pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
#                 Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
#                 If not provided, pooled text embeddings will be generated from `prompt` input argument.
#             negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
#                 Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
#                 weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
#                 input argument.
#             output_type (`str`, *optional*, defaults to `"pil"`):
#                 The output format of the generate image. Choose between
#                 [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
#             return_dict (`bool`, *optional*, defaults to `True`):
#                 Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
#                 of a plain tuple.
#             callback (`Callable`, *optional*):
#                 A function that will be called every `callback_steps` steps during inference. The function will be
#                 called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
#             callback_steps (`int`, *optional*, defaults to 1):
#                 The frequency at which the `callback` function will be called. If not specified, the callback will be
#                 called at every step.
#             cross_attention_kwargs (`dict`, *optional*):
#                 A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
#                 `self.processor` in
#                 [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
#             guidance_rescale (`float`, *optional*, defaults to 0.7):
#                 Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
#                 Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
#                 [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
#                 Guidance rescale factor should fix overexposure when using zero terminal SNR.
#             original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
#                 If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
#                 `original_size` defaults to `(width, height)` if not specified. Part of SDXL's micro-conditioning as
#                 explained in section 2.2 of
#                 [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
#             crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
#                 `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
#                 `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
#                 `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
#                 [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
#             target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
#                 For most cases, `target_size` should be set to the desired height and width of the generated image. If
#                 not specified it will default to `(width, height)`. Part of SDXL's micro-conditioning as explained in
#                 section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
#             negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
#                 To negatively condition the generation process based on a specific image resolution. Part of SDXL's
#                 micro-conditioning as explained in section 2.2 of
#                 [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
#                 information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
#             negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
#                 To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
#                 micro-conditioning as explained in section 2.2 of
#                 [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
#                 information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
#             negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
#                 To negatively condition the generation process based on a target image resolution. It should be as same
#                 as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
#                 [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
#                 information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
#             control_guidance_start (`float`, *optional*, defaults to 0.0):
#                 The percentage of total steps at which the ControlNet starts applying.
#             control_guidance_end (`float`, *optional*, defaults to 1.0):
#                 The percentage of total steps at which the ControlNet stops applying.

#         Examples:

#         Returns:
#             [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
#             [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
#             `tuple`. When returning a tuple, the first element is a list with the generated images.
#         """
#         # 0. Default height and width to unet
#         height = height or self.default_sample_size * self.vae_scale_factor
#         width = width or self.default_sample_size * self.vae_scale_factor

#         original_size = original_size or (height, width)
#         target_size = target_size or (height, width)

#         # 1. Check inputs. Raise error if not correct
#         self.check_inputs(
#             prompt,
#             prompt_2,
#             height,
#             width,
#             callback_steps,
#             negative_prompt,
#             negative_prompt_2,
#             prompt_embeds,
#             negative_prompt_embeds,
#             pooled_prompt_embeds,
#             negative_pooled_prompt_embeds,
#         )

#         # 2. Define call parameters
#         if prompt is not None and isinstance(prompt, str):
#             batch_size = 1
#         elif prompt is not None and isinstance(prompt, list):
#             batch_size = len(prompt)
#         else:
#             batch_size = prompt_embeds.shape[0]

#         device = self._execution_device

#         # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
#         # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
#         # corresponds to doing no classifier free guidance.
#         do_classifier_free_guidance = guidance_scale > 1.0

#         # 3. Encode input prompt
#         text_encoder_lora_scale = (
#             cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
#         )
#         (
#             prompt_embeds,
#             negative_prompt_embeds,
#             pooled_prompt_embeds,
#             negative_pooled_prompt_embeds,
#         ) = self.encode_prompt(
#             prompt=prompt,
#             prompt_2=prompt_2,
#             device=device,
#             num_images_per_prompt=num_images_per_prompt,
#             do_classifier_free_guidance=do_classifier_free_guidance,
#             negative_prompt=negative_prompt,
#             negative_prompt_2=negative_prompt_2,
#             prompt_embeds=prompt_embeds,
#             negative_prompt_embeds=negative_prompt_embeds,
#             pooled_prompt_embeds=pooled_prompt_embeds,
#             negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
#             lora_scale=text_encoder_lora_scale,
#         )

#         # 4. Prepare timesteps
#         self.scheduler.set_timesteps(num_inference_steps, device=device)

#         timesteps = self.scheduler.timesteps

#         # 5. Prepare latent variables
#         num_channels_latents = self.unet.config.in_channels
#         latents = self.prepare_latents(
#             batch_size * num_images_per_prompt,
#             num_channels_latents,
#             height,
#             width,
#             prompt_embeds.dtype,
#             device,
#             generator,
#             latents,
#         )

#         # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
#         extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

#         # 7. Prepare added time ids & embeddings
#         add_text_embeds = pooled_prompt_embeds
#         if self.text_encoder_2 is None:
#             text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
#         else:
#             text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

#         add_time_ids = self._get_add_time_ids(
#             original_size,
#             crops_coords_top_left,
#             target_size,
#             dtype=prompt_embeds.dtype,
#             text_encoder_projection_dim=text_encoder_projection_dim,
#         )
#         if negative_original_size is not None and negative_target_size is not None:
#             negative_add_time_ids = self._get_add_time_ids(
#                 negative_original_size,
#                 negative_crops_coords_top_left,
#                 negative_target_size,
#                 dtype=prompt_embeds.dtype,
#                 text_encoder_projection_dim=text_encoder_projection_dim,
#             )
#         else:
#             negative_add_time_ids = add_time_ids

#         if do_classifier_free_guidance:
#             prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
#             add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
#             add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

#         prompt_embeds = prompt_embeds.to(device)
#         add_text_embeds = add_text_embeds.to(device)
#         add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

#         # 8. Denoising loop
#         num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

#         # 7.1 Apply denoising_end
#         if denoising_end is not None and isinstance(denoising_end, float) and denoising_end > 0 and denoising_end < 1:
#             discrete_timestep_cutoff = int(
#                 round(
#                     self.scheduler.config.num_train_timesteps
#                     - (denoising_end * self.scheduler.config.num_train_timesteps)
#                 )
#             )
#             num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
#             timesteps = timesteps[:num_inference_steps]

#         # get init conditioning scale
#         for attn_processor in self.unet.attn_processors.values():
#             if isinstance(attn_processor, MOAttnProcessor_text):
#                 conditioning_scale = attn_processor.scale
#                 break


#         #NOTE:需要确认负面prompt应该怎么处理
#         # bsz = int(image_text_box_prompt_embeds.shape[0] / num_images_per_prompt)
#         # # image_text_box_prompt_embeds = torch.cat([torch.zeros_like(image_text_box_prompt_embeds),image_text_box_prompt_embeds],dim=0)
#         # # image_text_box_prompt_embeds = torch.cat([torch.zeros_like(image_text_box_prompt_embeds),torch.zeros_like(image_text_box_prompt_embeds)],dim=0)
#         # image_text_box_prompt_embeds = torch.cat([image_text_box_prompt_embeds,image_text_box_prompt_embeds],dim=0)

#         # prompt_embeds = (prompt_embeds, image_text_box_prompt_embeds)
#         # fusion_flag = True

#         with self.progress_bar(total=num_inference_steps) as progress_bar:
#             for i, t in enumerate(timesteps):

#                 #NOTE：实现方式比较粗暴，可以更优雅一些应该
#                 # if i >= mis*num_inference_steps and fusion_flag:
#                 #     latents = latents.view(num_images_per_prompt, bsz, latents.shape[-3], latents.shape[-2], latents.shape[-1])
#                 #     latents[:,0,:,:,:] = latents.mean(1).unsqueeze(1)
#                 #     latents = latents.view(-1,latents.shape[-3], latents.shape[-2], latents.shape[-1])

#                 #     fusion_flag = False

#                 if (i / len(timesteps) < control_guidance_start) or ((i + 1) / len(timesteps) > control_guidance_end):
#                     self.set_scale(0.0)
#                 else:
#                     self.set_scale(conditioning_scale)

#                 # expand the latents if we are doing classifier free guidance
#                 latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

#                 latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

#                 # predict the noise residual
#                 added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
#                 noise_pred = self.unet(
#                     latent_model_input,
#                     t,
#                     encoder_hidden_states=prompt_embeds,
#                     cross_attention_kwargs=cross_attention_kwargs,
#                     added_cond_kwargs=added_cond_kwargs,
#                     return_dict=False,
#                 )[0]

#                 # perform guidance
#                 if do_classifier_free_guidance:
#                     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

#                 if do_classifier_free_guidance and guidance_rescale > 0.0:
#                     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
#                     noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

#                 # compute the previous noisy sample x_t -> x_t-1
#                 latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

#                 # call the callback, if provided
#                 if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
#                     progress_bar.update()
#                     if callback is not None and i % callback_steps == 0:
#                         callback(i, t, latents)

#         if not output_type == "latent":
#             # make sure the VAE is in float32 mode, as it overflows in float16
#             needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
#             # print(self.vae.dtype)
#             # self.vae = self.vae.to(dtype=torch.float32)

#             if needs_upcasting:
#                 self.upcast_vae()
#                 latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            
#             latents=latents.to(torch.float32)
#             # print(latents.dtype)

#             image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
#             # print(self.vae.dtype)
#             # print(image.dtype)
#             # cast back to fp16 if needed
#             if needs_upcasting:
#                 self.vae.to(dtype=torch.float16)
#         else:
#             image = latents

#         if output_type != "latent":
#             # apply watermark if available
#             if self.watermark is not None:
#                 image = self.watermark.apply_watermark(image)

#             image = self.image_processor.postprocess(image, output_type=output_type)

#         # Offload all models
#         self.maybe_free_model_hooks()

#         if not return_dict:
#             return (image,)

#         return StableDiffusionXLPipelineOutput(images=image)






class StableDiffusionXLCustomPipeline(StableDiffusionXLPipeline):
    def set_scale(self, scale):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    @torch.no_grad()
    def __call__(  # noqa: C901
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(width, height)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(width, height)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            control_guidance_start (`float`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 7.1 Apply denoising_end
        if denoising_end is not None and isinstance(denoising_end, float) and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # get init conditioning scale
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                conditioning_scale = attn_processor.scale
                break

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if (i / len(timesteps) < control_guidance_start) or ((i + 1) / len(timesteps) > control_guidance_end):
                    self.set_scale(0.0)
                else:
                    self.set_scale(conditioning_scale)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if output_type != "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
