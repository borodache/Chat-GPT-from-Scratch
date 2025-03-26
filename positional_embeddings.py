import torch


from data_loading import create_dataloader_v1
from tokenizer import raw_text


vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)


max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
# print("Token IDs:\n", inputs)
# print("\nInputs shape:\n", inputs.shape)
token_embeddings = token_embedding_layer(inputs)
# print(token_embeddings)
# print(token_embeddings.shape)
# print("!!!")


context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings)
print(pos_embeddings.shape)
# print("***")
input_embeddings = token_embeddings + pos_embeddings
print(token_embeddings)
# print("---")
# print(input_embeddings)
# print(input_embeddings.shape)
