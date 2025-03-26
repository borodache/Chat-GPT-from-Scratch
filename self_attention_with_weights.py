import torch
import torch.nn as nn


from self_attention import input_chunks_embedding


# input_chunk_embedding_2 = input_chunks_embedding[1]
# d_in = input_chunks_embedding.shape[1]
# d_out = 2
#
# torch.manual_seed(123)
# W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
# W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
# W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
#
# query_2 = input_chunk_embedding_2 @ W_query
# key_2 = input_chunk_embedding_2 @ W_key
# value_2 = input_chunk_embedding_2 @ W_value
# print(query_2)
#
# keys = input_chunks_embedding @ W_key
# values = input_chunks_embedding @ W_value
# print("keys.shape:", keys.shape)
# print("values.shape:", values.shape)
#
# keys_2 = keys[1]
# attn_score_22 = query_2.dot(keys_2)
# print(attn_score_22)
#
# attn_scores_2 = query_2 @ keys.T
# print(attn_scores_2)
#
# print(keys)
# d_k = keys.shape[-1]
# print(d_k)
# # print("test softmax without normalization: ", torch.softmax(attn_scores_2, dim=-1))
# attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
# print(attn_weights_2)
#
# context_vec_2 = attn_weights_2 @ values
# print(context_vec_2)


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value
        attn_scores = queries @ keys.T  # omega
        print("attn_scores:", attn_scores)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        print("attn_weights:", attn_weights)
        context_vec = attn_weights @ values

        return context_vec


d_in = 3
d_out = 2


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec

# # torch.manual_seed(123)
# sa_v1 = SelfAttention_v1(d_in, d_out)
# print(sa_v1(input_chunks_embedding))
#
# torch.manual_seed(789)
# sa_v2 = SelfAttention_v2(d_in, d_out)
# print(sa_v2(input_chunks_embedding))


"""  Exercise 3.1  """


class SelfAttention_v3(nn.Module):
    def __init__(self, sa_v1: SelfAttention_v1):
        super().__init__()
        self.W_query = sa_v1.W_query
        self.W_key = sa_v1.W_key
        self.W_value = sa_v1.W_value

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec

# print("Excersice 3.1 results")
# torch.manual_seed(123)
# sa_v1 = SelfAttention_v1(d_in, d_out)
# print(sa_v1(input_chunks_embedding))
#
# torch.manual_seed(789)
# sa_v3 = SelfAttention_v3(sa_v1)
# print(sa_v3(input_chunks_embedding))
