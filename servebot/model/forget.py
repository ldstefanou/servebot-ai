import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForgettingAttention(nn.Module):
    def __init__(self, d_model, num_heads, forget_gate_type="full"):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.forget_gate_type = forget_gate_type

        # Standard QKV projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Forget gate - key innovation
        if forget_gate_type == "full":
            # Data-dependent forget gate
            self.forget_proj = nn.Linear(d_model, num_heads, bias=True)
        elif forget_gate_type == "bias_only":
            # Data-independent forget gate (learnable bias only)
            self.forget_bias = nn.Parameter(torch.zeros(num_heads))
        else:
            raise ValueError(f"Unknown forget_gate_type: {forget_gate_type}")

        # Compile the forward pass for speed
        self.forward = torch.compile(self._forward)

    def _forward(self, x, attention_mask=None):
        B, T, D = x.shape

        # Standard QKV computation
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        # Rearrange for attention: [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute forget gates
        if self.forget_gate_type == "full":
            # Data-dependent: f_t = sigmoid(W_f * x_t + b_f)
            forget_logits = self.forget_proj(x)  # [B, T, H]
            forget_logits = forget_logits.transpose(1, 2)  # [B, H, T]
            log_forget = F.logsigmoid(forget_logits)
        else:
            # Data-independent: f_t = sigmoid(b_f) (constant per head)
            forget_logits = self.forget_bias.view(1, self.num_heads, 1)
            forget_logits = forget_logits.expand(B, self.num_heads, T)
            log_forget = F.logsigmoid(forget_logits)

        # Core forgetting attention computation
        output = self._forgetting_attention_core(q, k, v, log_forget, attention_mask)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(output)

    def _forgetting_attention_core(self, q, k, v, log_forget, attention_mask=None):
        """
        Core forgetting attention mechanism

        Args:
            q, k, v: [B, H, T, D] query, key, value tensors
            log_forget: [B, H, T] log forget gates
            attention_mask: [B, H, T, T] or [B, T, T] optional attention mask
        """
        B, H, T, D = q.shape

        # Compute standard attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # [B, H, T, T]

        # Apply forget mechanism:
        # Cumulative forget decay: λ_i = Π_{k=1}^i f_k
        # Clamp log_forget to prevent extreme cumulative values
        log_forget = torch.clamp(log_forget, min=-10.0, max=0.0)
        cumsum_log_forget = torch.cumsum(log_forget, dim=-1)  # [B, H, T]

        # For position i attending to position j (j <= i):
        # decay_factor = λ_i / λ_j = exp(Σ_{k=j+1}^i log(f_k))
        decay_bias = cumsum_log_forget.unsqueeze(-1) - cumsum_log_forget.unsqueeze(
            -2
        )  # [B, H, T, T]

        # Apply causal mask (only attend to previous positions + self)
        causal_mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))

        # Apply optional attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 3:  # [B, T, T]
                attention_mask = attention_mask.unsqueeze(1)  # [B, 1, T, T]
            causal_mask = causal_mask & attention_mask

        # Clamp decay_bias BEFORE masking, then apply mask with large negative (not -inf)
        decay_bias = torch.clamp(decay_bias, min=-50.0, max=10.0)
        decay_bias = decay_bias.masked_fill(
            ~causal_mask, -1e9
        )  # Large negative instead of -inf

        # Modified attention: softmax(scores + decay_bias)
        # This downweights attention based on forget gates
        attention_weights = F.softmax(scores + decay_bias, dim=-1)

        # Apply attention to values
        output = torch.matmul(attention_weights, v)  # [B, H, T, D]

        return output


class ForgettingTransformerBlock(nn.Module):
    """Transformer block with forgetting attention"""

    def __init__(
        self, d_model, num_heads, d_ff=None, dropout=0.1, forget_gate_type="full"
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.attention = ForgettingAttention(d_model, num_heads, forget_gate_type)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Compile the forward pass for speed
        self.forward = torch.compile(self._forward)

    def _forward(self, x, attention_mask=None):
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x), attention_mask)
        x = x + attn_out

        # MLP with residual
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out

        return x
