from typing import List, Optional, Tuple, Union

import torch

if torch.cuda.get_device_capability()[0] >= 8:
    from flash_attn.ops.triton.layer_norm import RMSNorm
else:
    from torch.nn import RMSNorm

from deepspeed.ops.adam import FusedAdam
from deepspeed.runtime.activation_checkpointing.checkpointing import (
    checkpoint as checkpoint,
)
from eeg_datasets import EEGBatch
from einops import rearrange
from layers import MVPFormerGQAAttention, MVPFormerGQAFlashAttention
from loggers.patient_logger import PatientLogger, PredictiveLogger
from models import BrainDecider, BrainEncoder, BrainModel
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryStatScores
from torchtune.modules.peft import get_adapter_params, set_trainable_params
from transformers import (
    GPT2Config,
    GPT2Model,
    GPT2PreTrainedModel,
    get_cosine_schedule_with_warmup,
)
from transformers.models.gpt2.modeling_gpt2 import (
    BaseModelOutputWithPastAndCrossAttentions,
    GPT2Block,
)
from transformers.models.llama.modeling_llama import LlamaMLP


class MVPFormerConfig(GPT2Config):
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "max_channel_embeddings": "n_channels",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
        "intermediate_size": "n_inner",
        "hidden_act": "activation_function",
    }

    def __init__(
        self,
        n_positions=100,
        n_channels=128,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=2048,
        n_head_kv=12,
        global_att=True,
        activation_function="silu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=0.00001,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=False,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        lora=False,
        lora_alpha=16,
        lora_rank=8,
        lora_dropout=0,
        lora_merge=True,
    ):
        self.n_channels = n_channels
        self.global_att = global_att
        self.n_head_kv = n_head_kv
        self.pretraining_tp = 1
        self.mlp_bias = False
        self.lora = lora
        self.lora_alpha = lora_alpha
        self.lora_rank = lora_rank
        self.lora_dropout = lora_dropout
        self.lora_merge = lora_merge
        super().__init__(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            scale_attn_weights=scale_attn_weights,
            use_cache=use_cache,
            scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=reorder_and_upcast_attn,
            add_cross_attention=False,
        )


class MVPFormerBlock(GPT2Block):
    def __init__(self, config, layer_idx=None):
        torch.nn.Module.__init__(self)
        hidden_size = config.hidden_size
        self.ln_1 = RMSNorm(hidden_size, eps=config.layer_norm_epsilon)
        if torch.cuda.get_device_capability()[0] >= 8:
            self.attn = MVPFormerGQAFlashAttention(config, layer_idx=layer_idx)
        else:
            self.attn = MVPFormerGQAAttention(config, layer_idx=layer_idx)
        self.ln_2 = RMSNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = LlamaMLP(config)

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
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            positional_embedding,
            channel_embedding,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output + attn_output
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs  # hidden_states, present, (attentions, cross_attentions)


class MVPFormerModel(GPT2Model):
    def __init__(self, config, gradient_checkpointing=False):
        GPT2PreTrainedModel.__init__(self, config)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.n_head_kv
        self.head_dim = self.embed_dim // self.num_heads
        self.embed_kv_dim = self.head_dim * self.num_kv_heads
        self.drop = torch.nn.Dropout(config.embd_pdrop)
        self.h = torch.nn.ModuleList(
            [MVPFormerBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.ln_f = RMSNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.positional_embedding = torch.nn.Embedding(
            config.max_position_embeddings, self.embed_dim
        )
        self.channel_embedding = torch.nn.Embedding(
            config.max_channel_embeddings, self.embed_dim
        )
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = gradient_checkpointing
        self.post_init()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        channel_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        if inputs_embeds is None:
            raise NotImplementedError
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-2] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-2])
        if channel_ids is None:
            channel_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            channel_ids = channel_ids.unsqueeze(0).view(-1, input_shape[-1])
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        position_embeds = self.positional_embedding(position_ids)
        channel_embeds = self.channel_embedding(channel_ids)
        hidden_states = inputs_embeds
        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if layer_past is not None:
                    layer_past = tuple(
                        past_state.to(hidden_states.device) for past_state in layer_past
                    )
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                check_position_embeds = (
                    position_embeds.clone() if position_embeds is not None else None
                )
                check_channel_embeds = (
                    channel_embeds.clone() if channel_embeds is not None else None
                )
                check_attention_mask = (
                    attention_mask.clone() if attention_mask is not None else None
                )
                check_head_mask = (
                    head_mask[i].clone() if head_mask[i] is not None else None
                )
                check_encoder_hidden_states = (
                    encoder_hidden_states.clone()
                    if encoder_hidden_states is not None
                    else None
                )
                check_encoder_attention_mask = (
                    encoder_attention_mask.clone()
                    if encoder_attention_mask is not None
                    else None
                )
                outputs = checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    check_position_embeds,
                    check_channel_embeds,
                    None,
                    check_attention_mask,
                    check_head_mask,
                    check_encoder_hidden_states,
                    check_encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    position_embeds,
                    channel_embeds,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            if torch.is_tensor(outputs):
                outputs = (outputs,)
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (
                        outputs[3 if use_cache else 2],
                    )
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class HMVPFormer(BrainModel):
    def __init__(
        self,
        gpt_config: MVPFormerConfig,
        encoder: BrainEncoder,
        head: BrainDecider,
        num_negatives=20,
        temp=0.1,
        lr=0.1,
        warmup=None,
        training_steps=None,
        gradient_checkpointing: bool = False,
        base_model: str = "",
        head_model: str = "",
        save_output_tokens: bool = False,
        no_task: bool = False,
    ) -> None:
        super().__init__()
        self.save_output_tokens = save_output_tokens
        self.gpt_config = gpt_config
        self.mvpformer = MVPFormerModel(
            gpt_config, gradient_checkpointing=gradient_checkpointing
        )
        self.head = head
        self.mvpformer.set_input_embeddings(None)
        self.encoder = encoder
        self.loss = CrossEntropyLoss()
        self.num_negatives = num_negatives
        self.temp = temp
        self.lr = lr
        self.warmup = warmup
        self.training_steps = training_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.seizure_embeddings = torch.nn.Parameter(
            torch.normal(
                mean=0,
                std=1.0 / self.encoder.size_input,
                size=(3, self.encoder.size_input),
                device=self.device,
            )
        )
        self._patient_logger = None
        self._predictive_logger = None
        self._predictive_logger_random = None
        self._predictive_logger_twostep = None
        self.train_f1_score = BinaryF1Score()
        self.train_accuracy = BinaryAccuracy()
        self.train_accuracy_seizure = BinaryAccuracy()
        self.val_f1_score = BinaryF1Score()
        self.val_accuracy = BinaryAccuracy()
        self.val_accuracy_seizure = BinaryAccuracy()
        self.test_f1_score = BinaryF1Score()
        self.test_accuracy = BinaryAccuracy()
        self.test_accuracy_seizure = BinaryAccuracy()
        self.save_hyperparameters(ignore=["encoder", "base_model"])
        self.base_model = base_model
        self.no_task = no_task

        if self.gpt_config.lora:
            trainable_params = get_adapter_params(self)
            head_dict = {}
            for k, v in self.head.named_parameters():
                head_dict[f"head.{k}"] = v
            trainable_params.update(head_dict)
            set_trainable_params(self, trainable_params)
        if base_model != "" or head_model != "":
            try:
                base_weights = torch.load(base_model, weights_only=True)
            except FileNotFoundError:
                base_weights = None
            try:
                head_weights = torch.load(head_model, weights_only=True)
            except FileNotFoundError:
                head_weights = None
            self._load_checkpoint(base_weights, head_weights)

    def _load_checkpoint(self, base_weights, head_weights):
        if base_weights is not None:
            base_weights["encoder.ln.weight"] = torch.ones_like(self.encoder.ln.weight)
            self.load_state_dict(base_weights, strict=True)
            print("Base model loaded.")

    @property
    def patient_logger(
        self,
    ):
        if self._patient_logger is None:
            self._patient_logger = PatientLogger(
                save_dir=self.trainer.loggers[0].save_dir,
                name=self.trainer.loggers[0].name,
                version=self.trainer.loggers[0].version,
            )
        return self._patient_logger

    @property
    def predictive_logger(
        self,
    ):
        if self._predictive_logger is None:
            self._predictive_logger = PredictiveLogger(
                save_dir=self.trainer.loggers[0].save_dir,
                filename="sim",
                name=self.trainer.loggers[0].name,
                version=self.trainer.loggers[0].version,
            )
        return self._predictive_logger

    @property
    def predictive_logger_random(
        self,
    ):
        if self._predictive_logger_random is None:
            self._predictive_logger_random = PredictiveLogger(
                save_dir=self.trainer.loggers[0].save_dir,
                filename="sim_random",
                name=self.trainer.loggers[0].name,
                version=self.trainer.loggers[0].version,
            )
        return self._predictive_logger_random

    @property
    def predictive_logger_twostep(
        self,
    ):
        if self._predictive_logger_twostep is None:
            self._predictive_logger_twostep = PredictiveLogger(
                save_dir=self.trainer.loggers[0].save_dir,
                filename="sim_twostep",
                name=self.trainer.loggers[0].name,
                version=self.trainer.loggers[0].version,
            )
        return self._predictive_logger_twostep

    def forward(
        self,
        x: Tensor,
        past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tensor:
        x_flat = x.view(-1, x.shape[-1])
        seizure_flat = self.seizure_embeddings.view(
            -1, self.seizure_embeddings.shape[-1]
        )
        out_enc = self.encoder(torch.cat([x_flat, seizure_flat], dim=0)).squeeze()
        seizure_vectors = out_enc[-seizure_flat.shape[0] :].view(
            (*self.seizure_embeddings.shape[:-1], -1)
        )
        out_enc = out_enc[: -seizure_flat.shape[0]].view((*x.shape[:-1], -1))
        input_embeds = torch.concatenate(
            [
                out_enc[..., :-1, :],
                seizure_vectors[None, None, 2].expand(
                    out_enc.shape[0], out_enc.shape[1], -1, -1
                ),
            ],
            -2,
        )
        out = self.mvpformer(
            input_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        return out, out_enc, seizure_vectors

    @torch.compile()
    def _contrastive_logits(
        self,
        pred: torch.Tensor,
        positive: torch.Tensor,
        temp: float,
    ) -> torch.Tensor:
        negative = self._get_negative_samples(positive)[:, 1:]
        positive = positive[:, 1:]
        pred = pred[:, :-1]
        target_feat = torch.cat([positive.unsqueeze(-2), negative], dim=-2)
        logits = torch.cosine_similarity(
            pred.unsqueeze(-2), target_feat, dim=-1
        ).type_as(target_feat)
        with torch.no_grad():
            same_codes = (negative - positive.unsqueeze(-2)).norm(
                dim=-1
            ) < 1e-8 + 1e-5 * positive.norm(dim=-1)[..., None]
            logits[..., 1:][same_codes] = float("-inf")
        logits = logits / temp
        return logits.flatten(0, -2)

    def _common_forward(self, batch: Tensor) -> Tensor:
        x = batch  # x: [bsz, n_seg, ch, len]
        bsz, seg_n, ch, time = x.shape
        x_flat = rearrange(x, "bsz seg ch len -> (bsz seg ch) len")

        out_enc = self.encoder(x_flat)
        input_embeds = rearrange(
            out_enc, "(bsz seg ch) len -> bsz seg ch len", bsz=bsz, seg=seg_n, ch=ch
        )

        out = self.mvpformer(input_ids=None, inputs_embeds=input_embeds).last_hidden_state
        return out, input_embeds

    @staticmethod
    @torch.no_grad()
    def _merge_patients(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        y = torch.cat(y)
        max_channels = min([b.shape[2] for b in x])
        start_channel = [
            torch.randint(-1, b.shape[2] - max_channels, (1,), device=x[0].device)
            for b in x
        ]
        x = torch.cat(
            [
                b[
                    torch.arange(b.shape[0]),
                    :,
                    start_channel[i].clip(0) : start_channel[i].clip(0) + max_channels,
                ]
                for i, b in enumerate(x)
            ]
        )
        return x, y

    @torch.inference_mode()
    def _get_negative_samples(self, positive: Tensor) -> Tuple[Tensor, Tensor]:
        bsz, seg_n, ch, time = positive.shape
        random_idxs = torch.randint(
            (bsz - 1) * (ch * seg_n),
            (bsz * seg_n * ch * self.num_negatives,),
            device=self.device,
            dtype=torch.int,
        )
        candidates = (
            torch.arange(
                bsz * ch * seg_n,
                device=self.device,
            )
            .unsqueeze(-1)
            .expand(-1, self.num_negatives)
            .flatten()
        )
        random_idxs[random_idxs >= candidates] += 1
        negative = (
            positive.view(-1, time)
            .index_select(0, random_idxs.view(-1))
            .view(
                bsz,
                seg_n,
                ch,
                self.num_negatives,
                time,
            )
        )
        return negative

    @torch.inference_mode()
    def _compute_scores(
        self,
        x_shape: Tensor,
        logits: Tensor,
        y: Tensor,
        acc: BinaryAccuracy,
        acc_seiz: BinaryAccuracy,
        f1_score: BinaryF1Score,
    ):
        acc(
            logits.argmax(dim=1) == 0,
            torch.ones(logits.shape[0], device=logits.device),
        )

    def training_step(
        self,
        batch: EEGBatch,
        batch_idx: int,
    ) -> STEP_OUTPUT:
        x, y = batch.data, batch.label
        if isinstance(x, list):
            x, y = self._merge_patients(x, y)
        out, out_enc = self._common_forward(x)
        out = self.head(out)
        positive = out_enc
        logits = self._contrastive_logits(
            out,
            positive,
            self.temp,
        )
        targets = torch.zeros((logits.shape[0]), dtype=torch.long, device=self.device)
        loss = self.loss(logits, targets)
        self._compute_scores(
            x.shape,
            logits,
            y,
            self.train_accuracy,
            self.train_accuracy_seizure,
            self.train_f1_score,
        )
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_accuracy, prog_bar=True)
        # self.log("train/acc_seizure", self.train_accuracy_seizure)
        self.log(
            "train/learning_rate",
            self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0],
        )
        self.log(
            "train/f1_score",
            self.train_f1_score,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(
        self,
        batch: EEGBatch,
        batch_idx: int,
    ) -> STEP_OUTPUT:
        x, y = batch.data, batch.label
        if isinstance(x, list):
            x, y = self._merge_patients(x, y)
        out, out_enc, seizure_vectors = self._common_forward(x)
        pred = out[:, -1, ...].mean(dim=1)
        seizure_targets = seizure_vectors[None, :2].expand(out_enc.shape[0], -1, -1)
        compare = torch.nn.functional.cosine_similarity(
            pred.view(-1, 1, pred.shape[-1]), seizure_targets, dim=-1
        )
        pred_label = compare.argmax(dim=1)
        self.val_f1_score(pred_label, y.int().squeeze())
        self.val_accuracy_seizure(pred_label, y.int().squeeze())
        self.log(
            "val/f1_score",
            self.val_f1_score,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/acc_seizure",
            self.val_accuracy_seizure,
            on_epoch=True,
        )

    def test_step(
        self,
        batch: EEGBatch,
        batch_idx: int,
    ):
        x, y = batch.data, batch.label
        if isinstance(x, list):
            x, y = self._merge_patients(x, y)
        out, out_enc = self._common_forward(x)
        out = self.head(out)
        positive = out_enc
        out_cpu = out
        positive_onestep = positive[:, 1:]
        logits = torch.cosine_similarity(
            positive_onestep.unsqueeze(-2), out_cpu[:, :-1, ...].unsqueeze(-2), dim=-1
        ).type_as(out)
        positive_twostep = positive[:, 2:]
        logits_twostep = torch.cosine_similarity(
            positive_twostep.unsqueeze(-2), out_cpu[:, :-2, ...].unsqueeze(-2), dim=-1
        ).type_as(out)
        idx_random = torch.randint(out.shape[1], (1, out.shape[1] - 1)).flatten()
        idx_random[idx_random >= torch.arange(out.shape[1] - 1)] += 1
        idx_random = idx_random % out.shape[1] - 1
        positive_random = positive[:, idx_random]
        logits_random = torch.cosine_similarity(
            positive_random.unsqueeze(-2), out_cpu[:, :-1, ...].unsqueeze(-2), dim=-1
        ).type_as(out)
        self.predictive_logger.log(logits.detach().mean(dim=(2, 3)), batch.id[-1])
        self.predictive_logger_twostep.log(
            logits_twostep.detach().mean(dim=(2, 3)), batch.id[-1]
        )
        self.predictive_logger_random.log(
            logits_random.detach().mean(dim=(2, 3)), batch.id[-1]
        )

    def on_test_epoch_end(self) -> None:
        self.patient_logger.save()
        return

    def configure_optimizers(self):
        optimizer = FusedAdam(self.parameters(), lr=self.lr, weight_decay=0.01)
        if self.warmup is None:
            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
                optimizer=optimizer, lr_lambda=lambda _: 1.0
            )
        else:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup // self.trainer.accumulate_grad_batches,
                num_training_steps=self.training_steps
                // self.trainer.accumulate_grad_batches,
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class ClassificationHMVPFormer(HMVPFormer):
    def __init__(
        self,
        gpt_config: MVPFormerConfig,
        encoder: BrainEncoder,
        head: BrainDecider,
        num_negatives=20,
        temp=0.1,
        lr=0.1,
        warmup=None,
        training_steps=None,
        weight: Optional[List[float]] = None,
        gradient_checkpointing: bool = False,
        base_model: str = "",
        head_model: str = "",
        save_output_tokens: bool = False,
        no_task: bool = False,
    ) -> None:
        super().__init__(
            gpt_config,
            encoder,
            head,
            num_negatives,
            temp,
            lr,
            warmup,
            training_steps,
            gradient_checkpointing,
            base_model,
            head_model,
            save_output_tokens,
            no_task,
        )
        self.class_loss = CrossEntropyLoss(
            weight=None if weight is None else Tensor(weight)
        )
        self.test_bss = BinaryStatScores()

    def _load_checkpoint(self, base_weights, head_weights):
        if base_weights is not None:
            if base_weights["head.head.weight"].shape != self.head.head.weight.shape:
                del base_weights["head.head.weight"]
            miss_key, un_key = self.load_state_dict(base_weights, strict=False)

            if self.gpt_config.lora:
                non_trainable_missing_keys = [
                    value
                    for value in miss_key
                    if "lora" not in value.lower() and "head" not in value.lower()
                ]
                non_trainable_missing_keys = [
                    value
                    for value in non_trainable_missing_keys
                    if "encoder.proj_" not in value.lower()
                ]
                if len(non_trainable_missing_keys) > 0:
                    raise AttributeError(
                        f"Parameters {non_trainable_missing_keys} are missing from the base model and are not trainable."
                    )
                if len(un_key) > 0:
                    raise AttributeError(
                        f"Parameters {un_key} are unknown in the base model."
                    )
            else:
                if len(miss_key) > 0:
                    raise AttributeError(
                        f"Parameters {non_trainable_missing_keys} are missing from the base model."
                    )
                if len(un_key) > 0:
                    raise AttributeError(
                        f"Parameters {un_key} are unknown in the base model."
                    )

            print("Base model loaded.")

        if head_weights is not None:
            miss_key, un_key = self.load_state_dict(head_weights, strict=False)

            head_missing_keys = [
                value
                for value in miss_key
                if "head" in value.lower() or "lora" in value.lower()
            ]
            if not self.gpt_config.lora:
                head_missing_keys = [
                    value for value in head_missing_keys if "head" in value.lower()
                ]

            if len(head_missing_keys) > 0:
                raise AttributeError(
                    f"Parameters {head_missing_keys} are missing from the head model."
                )
            if len(un_key) > 0:
                if un_key != ["class_loss.weight"]:
                    raise AttributeError(
                        f"Parameters {un_key} are unknown in the model."
                    )
            print("Head model loaded.")

    def training_step(self, batch: EEGBatch, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch.data, batch.label
        if isinstance(x, list):
            x, y = self._merge_patients(x, y)
        out, _ = self._common_forward(x)
        out = out[:, -1].mean(dim=1)
        out = self.head(out)
        loss = self.class_loss(out, y.flatten().long())

        self.train_f1_score(out.argmax(dim=-1), y.flatten().long())
        self.log("train/loss", loss, prog_bar=True)
        self.log(
            "train/learning_rate",
            self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0],
        )
        self.log(
            "train/f1_score",
            self.train_f1_score,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch: EEGBatch, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch.data, batch.label
        if isinstance(x, list):
            x, y = self._merge_patients(x, y)
        out, _ = self._common_forward(x)
        out = out[:, -1].mean(dim=1)
        out = self.head(out)

        self.val_f1_score(out.argmax(dim=-1), y.flatten().long())
        self.log(
            "val/f1_score",
            self.val_f1_score,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

    def test_step(self, batch: EEGBatch, batch_idx: int) -> STEP_OUTPUT:
        x, y = batch.data, batch.label
        if isinstance(x, list):
            x, y = self._merge_patients(x, y)
        out, _ = self._common_forward(x)
        out = out[:, -1].mean(dim=1)
        out = self.head(out)

        self.test_f1_score(out.argmax(dim=-1), y.flatten().long())
        self.test_bss(out.argmax(dim=-1), y.flatten().long())
        self.patient_logger.log(out.detach(), batch.id[-1])
        self.log(
            "test/f1_score",
            self.test_f1_score,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        bss = self.test_bss.compute()
        self.log_dict(
            {
                "test/tp_epoch": bss[0],
                "test/fp_epoch": bss[1],
                "test/tn_epoch": bss[2],
                "test/fn_epoch": bss[3],
            }
        )


class MVPFormerHead(BrainDecider):
    def __init__(self, size: int, size_out: Optional[int] = None) -> None:
        super().__init__()
        self.head = torch.nn.Linear(
            size, size_out if size_out is not None else size, bias=False
        )

    def forward(self, x: torch.Tensor):
        out = self.head(x)

        return out
