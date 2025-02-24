from typing import List, Optional, Tuple, Union

import loralib as lora
import torch
from attentions import flashmvpa


class GeNIEGQAAttention(torch.nn.Module):
    def __init__(
        self,
        config,
        is_cross_attention=False,
        layer_idx=None,
    ):
        super().__init__()
        max_timesteps = config.max_position_embeddings
        self.register_buffer(
            "time_bias",
            torch.tril(
                torch.ones((max_timesteps, max_timesteps), dtype=torch.bool)
            ).view(1, 1, max_timesteps, max_timesteps),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.n_head_kv
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.embed_kv_dim = self.head_dim * self.num_kv_heads
        self.split_size = self.embed_dim
        self.inner_dim = config.n_inner
        self.global_att = config.global_att
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn
        if config.lora:
            self.q_attn = lora.Linear(
                self.embed_dim,
                self.embed_dim,
                lora_dropout=config.lora_dropout,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                bias=False,
                merge_weights=config.lora_merge,
            )
            self.c_attn = lora.MergedLinear(
                self.embed_dim,
                2 * self.embed_kv_dim,
                enable_lora=[False, True],
                lora_dropout=config.lora_dropout,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                bias=False,
                merge_weights=config.lora_merge,
            )  # W_q+W_Ein paper +V
        else:
            self.q_attn = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            self.c_attn = torch.nn.Linear(
                self.embed_dim,
                2 * self.embed_kv_dim,
                bias=False,
            )  # W_q+W_Ein paper +V
        self.position_net = torch.nn.Linear(
            self.embed_dim, self.embed_kv_dim, bias=False
        )  # W_{k,R}
        self.channel_net = torch.nn.Linear(
            self.embed_dim, self.embed_kv_dim, bias=False
        )  # W_{k,C}
        self.c_proj = torch.nn.Linear(
            self.embed_dim, self.embed_dim, bias=False
        )  # output net
        self.attn_bias = torch.nn.Parameter(
            torch.Tensor(3 * self.embed_kv_dim)
        )  # u+v+z in paper
        torch.nn.init.normal_(self.attn_bias, std=0.02)
        self.attn_dropout = torch.nn.Dropout(config.attn_pdrop)
        self.resid_dropout = torch.nn.Dropout(config.resid_pdrop)

    def adapter_params(self) -> List[str]:
        adapter_params = [
            "q_attn.lora_A",
            "q_attn.lora_B",
            "c_attn.lora_A",
            "c_attn.lora_B",
        ]
        return adapter_params

    @staticmethod
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, clen, tlen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :, :].expand(
            batch, num_key_value_heads, n_rep, clen, tlen, head_dim
        )
        return hidden_states.reshape(
            batch, num_key_value_heads * n_rep, clen, tlen, head_dim
        )

    @staticmethod
    def repeat_channel(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, head_dim, clen = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, :, None].expand(
            batch, num_key_value_heads, head_dim, clen, n_rep
        )
        return hidden_states.reshape(batch, num_key_value_heads, head_dim, clen * n_rep)

    @staticmethod
    def repeat_time(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, head_dim, tlen = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, None, :].expand(
            batch, num_key_value_heads, head_dim, n_rep, tlen
        )
        return hidden_states.reshape(batch, num_key_value_heads, head_dim, n_rep * tlen)

    @staticmethod
    def _rel_shift(x):
        zero_pad_shape = x.size()[:2] + (x.size(3), 1)
        x_review_shape = x.size()[:2] + (x.size(3), x.size(2))
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x.view(x_review_shape)], dim=-1)
        x_padded_shape = x.size()[:2] + (x.size(2) + 1, x.size(3))
        x_padded = x_padded.view(
            x_padded_shape[0], x_padded_shape[1], x_padded_shape[2], x_padded_shape[3]
        )
        x = x_padded[..., 1:, :].view_as(x)
        return x

    @staticmethod
    def _rel_shift_chan(x):
        chan_size = x.shape[-1]
        if chan_size > 1:
            upper_val = torch.cat(
                [
                    torch.arange(1, chan_size - i, dtype=torch.int32)
                    for i in range(chan_size - 1)
                ]
            )
        else:
            upper_val = torch.tensor([], dtype=torch.int32)
        idxes = torch.triu_indices(chan_size, chan_size, offset=1)
        shifting_idxes = torch.zeros(chan_size, chan_size, dtype=torch.int32)
        shifting_idxes[..., idxes[0], idxes[1]] = upper_val
        shifting_idxes.transpose(-2, -1)[..., idxes[0], idxes[1]] = upper_val
        shifting_idxes = (chan_size - 1 - shifting_idxes).repeat(
            x.shape[-2] // chan_size, 1
        )
        return x[..., torch.arange(x.size(-2)).unsqueeze(1), shifting_idxes]

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(
            0, 3, 1, 2, 4
        )  # (batch, head, seq_channels, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 3, 1, 4).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _rel_attn(
        self,
        query,
        global_key,
        time_key,
        channel_key,
        value,
        attention_mask=None,
        head_mask=None,
    ):
        bsz, _, pos_len, chan_len, _ = query.size()
        tot_len = pos_len * chan_len
        global_key_bias, time_key_bias, channel_key_bias = self.attn_bias.split(
            self.embed_kv_dim, dim=0
        )
        if self.global_att:
            global_key_bias = self._split_heads(
                global_key_bias[None, None, None, :], self.num_kv_heads, self.head_dim
            )
        time_key_bias = self._split_heads(
            time_key_bias[None, None, None, :], self.num_kv_heads, self.head_dim
        )
        channel_key_bias = self._split_heads(
            channel_key_bias[None, None, None, :], self.num_kv_heads, self.head_dim
        )
        if self.global_att:
            global_key_bias = self.repeat_kv(global_key_bias, self.num_kv_groups)
        channel_key_bias = self.repeat_kv(channel_key_bias, self.num_kv_groups)
        time_key_bias = self.repeat_kv(time_key_bias, self.num_kv_groups)
        if self.global_att:
            global_key = self.repeat_kv(global_key, self.num_kv_groups)
            global_key = global_key.reshape(
                bsz, self.num_heads, tot_len, self.head_dim
            ).transpose(-1, -2)
        time_key = self.repeat_kv(time_key, self.num_kv_groups)
        channel_key = self.repeat_kv(channel_key, self.num_kv_groups)
        time_key = time_key.squeeze(-2).transpose(-1, -2)
        channel_key = channel_key.squeeze(-3).transpose(-1, -2)
        if self.global_att:
            global_query_head = (query + global_key_bias).reshape(
                bsz, self.num_heads, -1, self.head_dim
            )
        time_query_head = (query + time_key_bias).reshape(
            bsz, self.num_heads, -1, self.head_dim
        )
        channel_query_head = (query + channel_key_bias).reshape(
            bsz, self.num_heads, -1, self.head_dim
        )
        if self.global_att:
            global_att = torch.matmul(global_query_head, global_key)
        time_att = torch.matmul(time_query_head, time_key)
        channel_att = torch.matmul(channel_query_head, channel_key)
        time_att = self._rel_shift(time_att)
        channel_att = self._rel_shift_chan(channel_att)
        attn_weights = self.repeat_channel(time_att, chan_len) + self.repeat_time(
            channel_att, pos_len
        )
        if self.global_att:
            window_mask = torch.logical_and(
                torch.tril(
                    torch.ones((pos_len, pos_len), device=query.device, dtype=bool),
                    diagonal=10,
                ),
                torch.triu(
                    torch.ones((pos_len, pos_len), device=query.device, dtype=bool),
                    diagonal=-10,
                ),
            )
            window_mask = window_mask.repeat_interleave(chan_len, 0).repeat_interleave(
                chan_len, 1
            )
            window_mask[-chan_len:] = 1
            attn_weights += global_att.masked_fill(~window_mask, 0.0)
        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [],
                value.size(-1) ** 0.5,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)
        if not self.is_cross_attention:
            causal_mask = (
                torch.tril(
                    torch.ones((pos_len, pos_len), device=query.device, dtype=bool)
                )
                .repeat_interleave(chan_len, 0)
                .repeat_interleave(chan_len, 1)
            )
            mask_value = torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights.masked_fill(~causal_mask, mask_value)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        if value.dtype == torch.float16:
            attn_weights = torch.nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.bfloat16
            ).to(value.dtype)
        else:
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        value = self.repeat_kv(value, self.num_kv_groups)
        attn_output = torch.matmul(attn_weights, value.flatten(2, 3))
        return attn_output.view_as(value), attn_weights

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        positional_embedding: Optional[Tuple[torch.FloatTensor]],
        channel_embedding: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(
                self.embed_kv_dim, dim=2
            )
            attention_mask = encoder_attention_mask
        else:
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(hidden_states).split(self.embed_kv_dim, dim=-1)
        time_key = self.position_net(positional_embedding).unsqueeze(2)
        channel_key = self.channel_net(channel_embedding).unsqueeze(1)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_kv_heads, self.head_dim)
        value = self._split_heads(value, self.num_kv_heads, self.head_dim)
        time_key = self._split_heads(time_key, self.num_kv_heads, self.head_dim)
        channel_key = self._split_heads(channel_key, self.num_kv_heads, self.head_dim)
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        if use_cache is True:
            present = (key, value)
        else:
            present = None
        attn_output, attn_weights = self._rel_attn(
            query,
            key,
            time_key,
            channel_key,
            value,
            attention_mask,
            head_mask,
        )
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs  # a, present, (attentions)


class GeNIEGQAFlashAttention(GeNIEGQAAttention):
    def forward(
        self,
        hidden_states: Tuple[torch.FloatTensor] | None,
        positional_embedding: Tuple[torch.FloatTensor] | None,
        channel_embedding: Tuple[torch.FloatTensor] | None,
        layer_past: Tuple[torch.Tensor] | None = None,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
    ) -> Tuple[torch.Tensor | Tuple[torch.Tensor], ...]:
        query = self.q_attn(hidden_states).contiguous()
        key, value = (
            self.c_attn(hidden_states).contiguous().split(self.embed_kv_dim, dim=-1)
        )
        time_key = self.position_net(positional_embedding).unsqueeze(2)
        channel_key = self.channel_net(channel_embedding).unsqueeze(1)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_kv_heads, self.head_dim)
        value = self._split_heads(value, self.num_kv_heads, self.head_dim)
        time_key = self._split_heads(time_key, self.num_kv_heads, self.head_dim)
        channel_key = self._split_heads(channel_key, self.num_kv_heads, self.head_dim)
        global_key_bias, time_key_bias, channel_key_bias = self.attn_bias.split(
            self.embed_kv_dim, dim=0
        )
        global_key_bias = self._split_heads(
            global_key_bias[None, None, None, :], self.num_kv_heads, self.head_dim
        )
        time_key_bias = self._split_heads(
            time_key_bias[None, None, None, :], self.num_kv_heads, self.head_dim
        )
        channel_key_bias = self._split_heads(
            channel_key_bias[None, None, None, :], self.num_kv_heads, self.head_dim
        )
        global_key_bias, time_key_bias, channel_key_bias = (
            global_key_bias.to(query.dtype),
            time_key_bias.to(query.dtype),
            channel_key_bias.to(query.dtype),
        )
        attn_output = flashmvpa(
            query.flatten(2, 3).contiguous(),
            global_key_bias.squeeze(2).contiguous(),
            time_key_bias.squeeze(2).contiguous(),
            channel_key_bias.squeeze(2).contiguous(),
            key.flatten(2, 3).contiguous(),
            time_key.flatten(2, 3).contiguous(),
            channel_key.flatten(2, 3).contiguous(),
            value.flatten(2, 3).contiguous(),
            causal=True,
            sm_scale=1.0 / (value.shape[-1] ** 0.5),
            window_size=10,
            p_drop=self.attn_dropout.p if self.training else 0,
            rel_bias_chan=True,
            rel_bias_time=True,
        )
        attn_output = attn_output.view_as(query).contiguous()
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        if use_cache is True:
            present = (key, value)
        else:
            present = None
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (None,)
        return outputs  # a, present, (attentions)
