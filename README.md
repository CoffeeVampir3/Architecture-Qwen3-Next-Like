Architecture:
- Deep seek style MoE (Auxillary loss free routing -- Different from Qwen's usual routing: https://arxiv.org/abs/2408.15664)
- Zero Centered RMS Norm /w Weight Decay (Concept from Qwen3-Next: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- Gated Attention (G1 per head variant specifically -- https://arxiv.org/abs/2505.06708)
- Gated Delta Net (Using FLA https://arxiv.org/abs/2412.06464)

Auxillary stuff:
- Cut cross entropy training (https://arxiv.org/abs/2411.09009)

### Do the thing
Using uv:
```
uv sync
```

Train (trains on 5% of TinyStories-hf for 10 epochs by default)
```
uv run python main.py
```

Infer (hard coded to use checkpoint 10):
```
uv run python basic_inf.py
```









