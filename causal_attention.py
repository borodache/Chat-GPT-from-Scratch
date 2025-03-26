from self_attention import input_chunks_embedding
# from self_attention_with_weights import SelfAttention_v1


from torch import nn
import torch


torch.manual_seed(123)

# class CasualAttention_v1(nn.Module):
#     def __init__(self, d_in, d_out, qkv_bias=False):
#         super().__init__()
#         self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
#         self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
#         self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
#
#     def forward(self, x):
#         keys = self.W_key(x)
#         queries = self.W_query(x)
#         values = self.W_value(x)
#         attn_scores = queries @ keys.T
#         print(attn_scores.shape)
#         attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
#         print("attn_weights:", attn_weights)
#         context_length = attn_scores.shape[0]
#         mask_simple = torch.tril(torch.ones(context_length, context_length))
#         masked_simple = attn_weights * mask_simple
#         print("masked_simple:", masked_simple)
#         row_sums = masked_simple.sum(dim=-1, keepdim=True)
#         masked_simple_norm = masked_simple / row_sums
#         print(masked_simple_norm)
#         context_vec = masked_simple_norm @ values
#
#         return context_vec


# ca_v1 = CasualAttention_v1(3, 2)
# print(ca_v1.forward(input_chunks_embedding))
torch.manual_seed(123)


# sa_v1 = SelfAttention_v1(3, 2)
# print(sa_v1.forward(input_chunks_embedding))
# print("***" * 25)


dropout = torch.nn.Dropout(0.5)


class CasualAttention_v2(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        print("attn_scores:", attn_scores)
        # print(attn_scores.shape)
        context_length = attn_scores.shape[0]
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        # print("mask:", mask)
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
        print("masked:", masked)
        attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
        print("attn_weights:", attn_weights)
        attn_weights = dropout(attn_weights)
        print("attn_weights after dropout:", attn_weights)
        context_vec = attn_weights @ values

        return context_vec


# ca_v2 = CasualAttention_v2(3, 2)
# print(ca_v2.forward(input_chunks_embedding))


""" Final Class taken from the book """
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values

        return context_vec


batch = torch.stack((input_chunks_embedding, input_chunks_embedding), dim=0)
# print(batch.shape)
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(3, 2, context_length, 0.0)
context_vecs = ca(batch)
# print("context_vecs.shape:", context_vecs.shape)
