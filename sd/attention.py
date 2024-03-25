import torch
from torch import nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embd: int, in_proj_bias: True, output_proj_bias: True):
        super().__init__()

        self.in_proj = nn.Linear(d_embd, 3 * d_embd, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embd, 3 * d_embd, bias = output_proj_bias)
        self.n_heads = n_heads
        self.d_heads = d_embd // n_heads

    def forward(self, x, causal_mask = False):

        input_shape = x.shape
        batch_size, seq_length, d_embd = input_shape

        interim_shape = (batch_size, seq_length, self.n_heads, self.d_heads)

        q, k, v = self.in_proj(x).chunk(3, dim = -1)

        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)

        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype = torch.bool).triu(1)
            weight.masked_fill(mask, -torch.inf)

        weight /= math.sqrt(self.d_heads)

        weight = F.softmax(weight, dim = -1)

        output = weight @ v
        
        output = output.transpose(1,2) 

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embd: int, d_cross: int, in_proj_bias: True, output_proj_bias: True):
        super().__init__()
        self.q_proj = nn.Linear(d_embd, d_embd, bias = in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embd, bias = in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embd, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embd, d_embd, bias = output_proj_bias)
        self.n_heads = n_heads
        self.d_heads = d_embd // n_heads

    def forward(self, x, y):

        input_shape = x.shape
        batch_size, seq_length, d_embd = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_heads)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)

        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.d_heads)

        weight = F.softmax(weight, dim = -1)

        output = weight @ v
        
        output = output.transpose(1,2).contiguous()

        output = output.view(input_shape)

        output = self.out_proj(output)

        return output
