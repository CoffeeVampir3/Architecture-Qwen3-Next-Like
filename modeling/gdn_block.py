import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.layers import GatedDeltaNet

from .q3_deltanet import Qwen3NextGatedDeltaNet
from .moe_layer import MoELayer
from .zRMSNorm import ZeroCenteredRMSNorm

class GatedDeltaNetBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.input_layernorm = ZeroCenteredRMSNorm(config.hidden_size)
        self.gated_deltanet = Qwen3NextGatedDeltaNet(config, layer_idx)
        self.post_attention_layernorm = ZeroCenteredRMSNorm(config.hidden_size)
        self.mlp = MoELayer(config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # Input shape: [batch_size, seq_len, hidden_size]
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)  # [batch_size, seq_len, hidden_size]
        hidden_states = self.gated_deltanet(hidden_states)  # [batch_size, seq_len, hidden_size]
        hidden_states = residual + hidden_states  # [batch_size, seq_len, hidden_size]

        # MoE sublayer
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, topk_idx = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, topk_idx
