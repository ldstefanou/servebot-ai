import math

import torch
import torch.nn as nn


class ContinuousTimeEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.time_scales = [1, 7, 30, 365, 365 * 4, 365 * 10]
        self.dims_per_scale = d_model // len(self.time_scales)

    def forward(self, timestamps):
        batch_size, seq_len = timestamps.shape

        # Convert to relative time
        target_time = timestamps[:, -1:]
        relative_days = (timestamps - target_time).float() / (60 * 60 * 24)

        encodings = []

        for scale in self.time_scales:
            scale_dims = self.dims_per_scale

            # Make sure we have even number of dimensions for sin/cos pairs
            if scale_dims % 2 != 0:
                scale_dims -= 1

            # Create frequency terms
            freqs = torch.exp(
                torch.arange(0, scale_dims // 2).float()
                * -(math.log(10000.0) / (scale_dims // 2))
            )

            # Scale time
            scaled_time = relative_days.unsqueeze(-1) / scale

            # Create encoding with exact dimensions
            scale_encoding = torch.zeros(
                batch_size, seq_len, scale_dims, device=timestamps.device
            )

            # Apply sin/cos to pairs
            sin_part = torch.sin(scaled_time * freqs)
            cos_part = torch.cos(scaled_time * freqs)

            scale_encoding[:, :, 0::2] = sin_part
            scale_encoding[:, :, 1::2] = cos_part

            encodings.append(scale_encoding)

        # Concatenate and pad to exact d_model size
        full_encoding = torch.cat(encodings, dim=-1)

        if full_encoding.size(-1) < self.d_model:
            padding = torch.zeros(
                batch_size,
                seq_len,
                self.d_model - full_encoding.size(-1),
                device=timestamps.device,
            )
            full_encoding = torch.cat([full_encoding, padding], dim=-1)

        return full_encoding[:, :, : self.d_model]
