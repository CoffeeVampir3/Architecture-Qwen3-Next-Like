from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    embed_size: int = 256
    hidden_size: int = 256
    intermediate_size: int = 512

    transformer_depth: int = 4

    n_experts: int = 48
    n_shared_experts: int = 16
    n_routed_experts: int = n_experts - n_shared_experts
    n_experts_per_token = 4

    n_attention_heads: int = 16
    n_key_value_heads: int = 4

                self.num_v_heads = 32
                self.num_k_heads = 16
                self.head_k_dim = 128
                self.head_v_dim = 128

    num_delta_v_heads = 32
    num_delta_k_heads = 16
    v_head_dim = 128
    k_head_dim = 128

    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 8192
    rope_theta: int = 100000
