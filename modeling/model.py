import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_block import TransformerBlock
from .gdn_block import GatedDeltaNetBlock

class MoEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_size)

        self.layers = nn.ModuleList(
            [TransformerBlock(config)] +
            [GatedDeltaNetBlock(config, layer_idx=i) for i in range(config.transformer_depth - 1)]
        )

        self.output_layer = nn.Linear(config.embed_size, config.vocab_size)

    def forward(self, x):
        x = self.embedding(x)

        all_topk_indices = []
        for layer in self.layers:
            x, topk_idx = layer(x)
            all_topk_indices.append(topk_idx)

        x = self.output_layer(x)

        return x, all_topk_indices

    # To support CCE type of loss.
    def headless_forward(self, x):
        x = self.embedding(x)

        all_topk_indices = []
        for layer in self.layers:
            x, topk_idx = layer(x)
            all_topk_indices.append(topk_idx)

        return x, all_topk_indices

    def get_classifier_weights(self):
        return self.output_layer.weight
