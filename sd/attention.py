import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        # usually embeddings capture information about the 'token'
        # here, the features/channels for a pixel capture that information
        # d_embd <==> channels

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias = in_proj_bias)
        self.out_bias = nn.Linear(d_embed, d_embed, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask = False):
        # x: (Batch_Size, Seq_len, Dim)
        # Coming from to x: (Batch_Size, Height * Width, channels)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
     
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim*3) -> 3 tensors of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim = -1)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        # NOTE: Observe how this is beautiful because for multi heads, we need seq_len * head_dim shaped vectors only
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        
        # (Batch_Size, H, Seq_Len, Seq_Len)
        # (Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Mask where upper triangle is 1s
            # NOTE: Always remember we apply mask BEFORE softmax
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)

            weight.masked_fill(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim = -1)

        # No transpose? Looks like last of weight is consumed by second-last of v (Seq_Len)
        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v

        # (Batch_Size, H, Seq_Len, Dim / H) ->  (Batch_Size, Seq_Len, H , Dim / H)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (Batch_Size, Seq_Len, Dim)
        return output 


   
class CrossAttention(nn.Module):
    # very similar to SelfAttention except that query comes from a different place than keys and values 
    # i.e. CrossAttention :D

    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias = True, out_proj_bias = True):
        # d_embed is for query
        # d_cross is for keys and values
        super().__init__()
        self.q_proj = nn.Linear(in_features=d_embed, out_features=d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(in_features=d_cross, out_features=d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(in_features=d_cross, out_features=d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(in_features=d_embed, out_features=d_embed, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, y):
        # x: (latent): (Batch_Size, Seq_Len_Q, Dim_Q)
        # y: (context): (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Multiply by Wq
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x)

        # (Batch_Size, Seq_Len_Q, Dim_KV) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, d_head) -> (Batch_Size, H, Seq_Len, d_head)
        # NOTE: Again, observe beauty of transpose
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # similar as in SelfAttention
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)

        # NOTE: no causal mask in cross attention because we are relating the prompt with the pixels. Anything in the prompt is allowed to see anything in the pixels
        weight = F.softmax(weight, dim = -1)

        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)

        return output

