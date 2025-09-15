# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn

# proposed method
class MUSE_AttnProcessor(nn.Module):
    r"""
    Attention processor for MUSE (Multi-Subject Unified Synthesis via Explicit Layout Semantic Expansion).
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        text_in_dim (`int`):
            The size of the text grounding token.
        image_in_dim (`int`):
            The size of the image grounding token.
        scale (`float`, defaults to 1.0):
            the weight scale of subject personalization.
        num_tokens_text (`int`, defaults to 1):
            The context length of the text features.
        num_tokens_image (`int`, defaults to 1 when apply resampler it should be 4):
            The context length of the image features.
        num_tokens_text (`int`, defaults to 10):
            The number of the subjects.
    """

    def __init__(
        self, 
        hidden_size, 
        text_in_dim=768,
        image_in_dim=2048, 
        scale=1.0, 
        num_object=10,
        num_tokens_text=1,
        num_tokens_image=1,
        stage=1
    ):

        super().__init__()

        self.hidden_size = hidden_size
        self.scale = scale
        self.num_object = num_object
        self.num_tokens_text = num_tokens_text
        self.num_tokens_image = num_tokens_image
        self.stage = stage

        self.to_k_text = nn.Linear(text_in_dim, hidden_size, bias=False)
        self.to_v_text = nn.Linear(text_in_dim, hidden_size, bias=False)
        
        if self.stage==2:
            self.to_k_text.requires_grad_(False)
            self.to_v_text.requires_grad_(False)
            
            self.to_k_image = nn.Linear(image_in_dim or hidden_size, hidden_size, bias=False)
            self.to_v_image = nn.Linear(image_in_dim or hidden_size, hidden_size, bias=False)


    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        text_box_grounding_embeds = None,
        image_box_grounding_embeds = None,
    ):

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            hidden_states_grounding_text = text_box_grounding_embeds
            hidden_states_grounding_image = image_box_grounding_embeds

            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)


        #-------------cross attn
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        # attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        #-------------concatenated cross attn
        if text_box_grounding_embeds is not None:
            key_text = self.to_k_text(hidden_states_grounding_text)
            value_text = self.to_v_text(hidden_states_grounding_text)

            key = torch.cat([key, key_text],dim=-2)
            value = torch.cat([value, value_text], dim=-2)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        #-------------decoupled cross attn
        if self.stage == 2 and image_box_grounding_embeds is not None:
            key_image = self.to_k_image(hidden_states_grounding_image)
            value_image = self.to_v_image(hidden_states_grounding_image)
            key_image = attn.head_to_batch_dim(key_image)
            value_image = attn.head_to_batch_dim(value_image)

            attention_probs_image = attn.get_attention_scores(query, key_image, attention_mask)
            hidden_states_grounding_image = torch.bmm(attention_probs_image, value_image)
            hidden_states_grounding_image = attn.batch_to_head_dim(hidden_states_grounding_image)

            hidden_states = hidden_states + self.scale * hidden_states_grounding_image

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states  +residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states




class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states