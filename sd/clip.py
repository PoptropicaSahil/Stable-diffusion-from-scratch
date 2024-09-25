import torch
from torch import nn
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_tokens: int):
        super().__init__()
        # n_embd <==> Dim

        # Note how these are lookup tables for n_vocab and n_tokens respectively
        # Positional embedding is LEARNT here
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))

    def forward(self, tokens):
        # (Batch_Size, Seq_Len) -> # (Batch_Size, Seq_Len, Dim)
        x = self.token_embedding(tokens)
        x += self.position_embedding
        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.self_attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Seq_Len, Dim)
        # Encoder-only model - check the image in readme

        residue = x

        # SelfAttention
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim) in every step
        x = self.layernorm_1(x)        
        x = self.self_attention(x, causal_mask=True)
        x += residue

        residue = x

        # Feedforward
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_2(x)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 4*Dim)
        x = self.linear_1(x)

        # (Batch_Size, Seq_Len, 4*Dim) -> (Batch_Size, Seq_Len, 4*Dim)
        x = x * torch.sigmoid(1.702 * x)  # QuickGELU activation function

        # (Batch_Size, Seq_Len, 4*Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.linear_2(x)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        return x


class CLIP(nn.Module):
    # CLIP is seq2seq model so output shape should match input shape
    def __init__(self):
        # n_tokens equivalent to max_seq_len
        self.embedding = CLIPEmbedding(n_vocab=49408, n_embd=768, n_tokens=77)

        self.layers = nn.ModuleList(
            [CLIPLayer(n_head=12, n_embd=768) for _ in range(12)]
        )

        self.layernorm = nn.LayerNorm(768)


    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # tokens: (Batch_Size, Seq_Len)
        
        # NOTE: check why LongTensor 
        # - maybe because pos embedd requires more precision

        tokens = tokens.type(torch.long)  # type: ignore

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)


        # (Batch_Size, Seq_Len, Dim)
        output = self.layernorm(state)

        return output

