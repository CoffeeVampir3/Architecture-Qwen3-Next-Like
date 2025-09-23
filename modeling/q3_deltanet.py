import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fla.modules import FusedRMSNormGated
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

from .model_config import ModelConfig

# Large portions from transformers: //github.com/huggingface/transformers/blob/d42e96a2a731c4a772e396baa0d915524c873ff0/src/transformers/models/qwen3_next/modeling_qwen3_next.py
# It's been rewritten to use einops notation where possible as the original was hard for me to understand.

def apply_mask_to_padding_states(hidden_states, attention_mask):
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        mask_expanded = attention_mask.unsqueeze(-1)
        hidden_states = (hidden_states * mask_expanded).to(dtype)
    return hidden_states

def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = 1 / torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

class DeltaNetStateCache:
    def __init__(self):
        self.conv_states = {}
        self.recurrent_states = {}

    @property
    def has_previous_state(self) -> bool:
        return any(state is not None for state in self.conv_states.values())

class Qwen3NextGatedDeltaNet(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.num_delta_v_heads
        self.num_k_heads = config.num_delta_k_heads
        self.head_k_dim = config.k_head_dim
        self.head_v_dim = config.v_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = 4
        self.layer_idx = layer_idx
        self.activation = "silu"
        self.act = F.silu
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # projection of the input hidden states
        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2
        self.in_proj_qkvz = nn.Linear(self.hidden_size, projection_size_qkvz, bias=False)
        self.in_proj_ba = nn.Linear(self.hidden_size, projection_size_ba, bias=False)

        # time step projection (discretization)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = (
                FusedRMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                activation=self.activation,
                device=torch.cuda.current_device(),
                dtype=torch.float32
            )
        )

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update
        self.chunk_gated_delta_rule = chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule

        self.reset_parameters()

    def reset_parameters(self):
        bound = 0.25

        nn.init.uniform_(self.out_proj.weight, -0.06, 0.06)
        nn.init.uniform_(self.in_proj_qkvz.weight, -bound, bound)
        nn.init.uniform_(self.in_proj_ba.weight, -bound, bound)

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        v_heads_per_k_head = self.num_v_heads // self.num_k_heads
        value_dim_per_k_head = v_heads_per_k_head * self.head_v_dim

        # Unpack heads and features
        mixed_qkvz = rearrange(
            mixed_qkvz,
            '... (num_k_heads features_per_head) -> ... num_k_heads features_per_head',
            num_k_heads=self.num_k_heads
        )
        mixed_ba = rearrange(
            mixed_ba,
            '... (num_k_heads features_per_head) -> ... num_k_heads features_per_head',
            num_k_heads=self.num_k_heads
        )

        # Unpack from (qkvz) -> q, k, v, z
        query, key, value, z = torch.split(
            mixed_qkvz,
            [self.head_k_dim, self.head_k_dim, value_dim_per_k_head, value_dim_per_k_head],
            dim=-1
        )

        # (b, a) -> b, a
        b, a = torch.split(
            mixed_ba,
            [v_heads_per_k_head, v_heads_per_k_head],
            dim=-1
        )

        value = rearrange(
            value,
            '... num_k_heads (v_per_k head_v_dim) -> ... (num_k_heads v_per_k) head_v_dim',
            v_per_k=v_heads_per_k_head, head_v_dim=self.head_v_dim
        )
        z = rearrange(
            z,
            '... num_k_heads (v_per_k head_v_dim) -> ... (num_k_heads v_per_k) head_v_dim',
            v_per_k=v_heads_per_k_head, head_v_dim=self.head_v_dim
        )
        b = rearrange(b, '... num_k_heads v_per_k -> ... (num_k_heads v_per_k)')
        a = rearrange(a, '... num_k_heads v_per_k -> ... (num_k_heads v_per_k)')
        return query, key, value, z, b, a

    def forward(
        self,
        hidden_states,
        cache_params = None,
        attention_mask = None,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
        )

        if cache_params is not None:
            conv_state = cache_params.conv_states.get(self.layer_idx)
            recurrent_state = cache_params.recurrent_states.get(self.layer_idx)

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query = rearrange(query, 'batch seq heads head_dim -> batch seq (heads head_dim)')
        key = rearrange(key, 'batch seq heads head_dim -> batch seq (heads head_dim)')
        value = rearrange(value, 'batch seq heads head_dim -> batch seq (heads head_dim)')

        mixed_qkv = rearrange(
            torch.cat([query, key, value], dim=-1),
            'batch seq features -> batch features seq'
        )

        if use_precomputed_states:
            # 2. Convolution sequence transformation
            # NOTE: the conv state is updated in `causal_conv1d_update`
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = rearrange(
            mixed_qkv,
            'batch features seq -> batch seq features'
        )

        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )

        query = rearrange(
            query,
            'batch seq (num_heads head_dim) -> batch seq num_heads head_dim',
            head_dim=self.head_k_dim
        )

        key = rearrange(
            key,
            'batch seq (num_heads head_dim) -> batch seq num_heads head_dim',
            head_dim=self.head_k_dim
        )

        value = rearrange(
            value,
            'batch seq (num_heads head_dim) -> batch seq num_heads head_dim',
            head_dim=self.head_v_dim
        )

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        core_attn_out = rearrange(core_attn_out, 'batch seq heads head_dim -> (batch seq heads) head_dim')
        z_flat = rearrange(z, 'batch seq heads head_dim -> (batch seq heads) head_dim')

        core_attn_out = self.norm(core_attn_out, z_flat)

        core_attn_out = rearrange(
            core_attn_out,
            '(batch seq heads) head_dim -> batch seq (heads head_dim)',
            batch=z.shape[0],
            seq=z.shape[1],
            heads=z.shape[2]
        )

        output = self.out_proj(core_attn_out)
        return output
