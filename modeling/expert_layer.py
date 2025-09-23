import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.gate_proj.weight, -0.06, 0.06)
        nn.init.uniform_(self.up_proj.weight, -0.06, 0.06)
        nn.init.uniform_(self.down_proj.weight, -0.06, 0.06)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
