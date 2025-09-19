import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

from .moe_gate import MoEGate
from .expert_layer import ExpertMLP

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts_per_token = config.n_experts_per_token
        self.experts = nn.ModuleList([
            ExpertMLP(config, intermediate_size=config.intermediate_size)
            for i in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        intermediate_size = config.intermediate_size * config.n_shared_experts
        self.shared_experts = ExpertMLP(config, intermediate_size=intermediate_size)

    def forward(self, hidden_states):
        identity = hidden_states
        topk_idx, topk_weight = self.gate(hidden_states)

        shared_output = self.shared_experts(identity)

        routed_expert_outputs = torch.stack([expert(hidden_states) for expert in self.experts])
        expert_selection = F.one_hot(topk_idx, num_classes=self.config.n_routed_experts).to(hidden_states.dtype) * topk_weight.unsqueeze(-1)
        routed_output = einsum(expert_selection, routed_expert_outputs,
                            'batch seq k experts, experts batch seq hidden -> batch seq hidden')

        final_output = shared_output + routed_output

        return final_output, topk_idx
