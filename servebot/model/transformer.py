from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, seq_length: int):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(seq_length, d_model)
        self.register_buffer(
            "scale", torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = token_ids.shape
        positions = (
            torch.arange(seq_length, device=token_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        token_emb = self.token_embedding(token_ids)
        pos_emb = self.pos_embedding(positions)

        return (token_emb + pos_emb) * self.scale


class AttentionHead(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.qkv = nn.Linear(config["d_model"], config["d_attention"] * 3, bias=False)

    def forward(self, x, attn_mask=None):
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Create causal mask
        seq_len = x.size(1)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )

        # Combine with attention mask if provided
        if attn_mask is None:
            attn_mask = causal_mask

        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)


class MLP(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["d_model"], config["d_hidden"]),
            nn.GELU(),
            nn.Linear(config["d_hidden"], config["d_model"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, block_config: Dict[str, Any], global_config: Dict[str, Any]):
        super().__init__()

        # Get number of heads for this block
        n_heads = block_config.get("n_heads", 4)
        d_model = global_config["d_model"]
        d_attention = d_model // n_heads

        # Build attention heads
        head_config = {"d_model": d_model, "d_attention": d_attention}
        self.heads = nn.ModuleList([AttentionHead(head_config) for _ in range(n_heads)])

        # Build MLP
        if "mlp" in block_config:
            mlp_config = {**global_config, **block_config["mlp"]}
            self.mlp = MLP(mlp_config)
        else:
            self.mlp = MLP(global_config)

        self.proj = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(global_config.get("dropout", 0.1))

    def forward(self, x, attn_mask=None):
        # Multi-head attention
        head_outputs = [head(x, attn_mask=attn_mask) for head in self.heads]
        att_out = torch.cat(head_outputs, dim=-1)
        att_out = self.proj(att_out)

        out = x + self.dropout(att_out)
        out = self.ln1(out)

        # MLP
        mlp_out = self.mlp(out)
        out = out + self.dropout(mlp_out)
        out = self.ln2(out)

        return out


class Transformer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # Extract global config
        global_config = {k: v for k, v in config.items() if not k.startswith("block_")}

        # Build blocks dynamically
        self.blocks = nn.ModuleList()
        for key, block_config in config.items():
            if key.startswith("block_"):
                self.blocks.append(TransformerBlock(block_config, global_config))

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)
