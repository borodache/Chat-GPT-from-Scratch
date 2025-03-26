import torch


"""
Self-Attention for one chunk (query)
"""

# words embeddings (positional)
input_chunks_embedding = torch.tensor([[0.43, 0.15, 0.89],  # Your (x^1)
                                       [0.55, 0.87, 0.66],  # journey (x^2)
                                       [0.57, 0.85, 0.64],  # starts (x^3)
                                       [0.22, 0.58, 0.33],  # with (x^4)
                                       [0.77, 0.25, 0.10],  # one (x^5)
                                       [0.05, 0.80, 0.55]]  # step (x^6)
                                      )

query = input_chunks_embedding[1]
attn_scores_2 = torch.empty(input_chunks_embedding.shape[0])
for i, chunk_embedding_i in enumerate(input_chunks_embedding):
    attn_scores_2[i] = torch.dot(chunk_embedding_i, query)

# print(attn_scores_2)

# attention weights of chunk two after softmax normalization
attn_weights_2_naive = torch.softmax(attn_scores_2, dim=0)
# attn_weights_2_naive = torch.tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
# print(attn_weights_2_naive)

# query = input_chunks_embedding[1]
# print(query)
context_vec_2 = torch.zeros(query.shape)
for i, chunk_embedding_i in enumerate(input_chunks_embedding):
    context_vec_2 += attn_weights_2_naive[i] * chunk_embedding_i

# print(context_vec_2)


"""
Self-Attention for all input chunks (queries)
"""

# attn_scores = torch.empty(6, 6)
# for i, chunk_embedding_i in enumerate(input_chunks_embedding):
#     for j, x_j in enumerate(input_chunks_embedding):
#         attn_scores[i, j] = torch.dot(chunk_embedding_i, x_j)

attn_scores = input_chunks_embedding @ input_chunks_embedding.T

# print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim = 1)
# print(attn_weights)

# context_vecs = torch.zeros(input_chunks_embedding.shape)
# for i, chunk_embedding_i in enumerate(input_chunks_embedding):
#     context_vec += attn_weights_2_naive[i] * chunk_embedding_i
context_vecs = attn_weights @ input_chunks_embedding
# print(context_vecs)
