import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


from tokenizer import raw_text


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers
                            )

    return dataloader


dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
# data_iter = iter(dataloader)
# first_batch = next(data_iter)
# print(first_batch)
# second_batch = next(data_iter)
# print(second_batch)

# # Exercise 2.2
# # max_length=2 and stride=2
# dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=2, stride=2, shuffle=False)
# data_iter = iter(dataloader)
# curr_data = next(data_iter)
# i = 0
#
# while curr_data:
#     print(curr_data)
#     curr_data = next(data_iter)
#     i += 1
#     if i > 10:
#         break
#
# print("****")
# # max_length=8 and stride=2
# dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=8, stride=2, shuffle=False)
# data_iter = iter(dataloader)
# curr_data = next(data_iter)
# i = 0
#
# while curr_data:
#     print(curr_data)
#     curr_data = next(data_iter)
#     i += 1
#     if i > 10:
#         break
#
# # print("****")
# print("Finished!")


dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
# print("Inputs:\n", inputs)
# print("\nTargets:\n", targets)
