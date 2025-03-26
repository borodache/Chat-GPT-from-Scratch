import re

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

vocab = dict(zip(list(all_words), range(vocab_size)))


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]

        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # print("1)", text)
        text = re.sub(r'\s+([,.?!()])', r'\1', text)
        # print("2)", text)
        text = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', text) # remove un relevant spaces between two double qoutes
        # print("3)", text)
        text = re.sub(r' \' s ', "'s ", text)  # make the qoute and s of possesion (it's)
        # print("4)", text)
        text = re.sub(r'\'\s([^\']*?)\s\'', r"'\1'", text)  # remove un relevant spaces between two single qoutes
        # print("5)", text)

        return text


# tokenizer = SimpleTokenizerV1(vocab)
# text = """"It's the last 'he painted, you know,'" Mrs. Gisburn said with pardonable pride."""
# print(text)
# ids = tokenizer.encode(text)
# # print(ids)
# # print(tokenizer.decode(ids))
# print(tokenizer.decode(ids) == text)
#
# text2 = "Hello, do you like tea?"
# print(tokenizer.encode(text2))


all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
new_vocab_size = len(all_tokens)

new_vocab = dict(zip(list(all_tokens), range(new_vocab_size)))

# print(len(vocab.items()))
# for i, item in enumerate(list(vocab.items())[-5:]):
#     print(item)


class SimpleTokenizerV2:
    def __init__(self, new_vocab):
        self.str_to_int = new_vocab
        self.int_to_str = {i: s for s, i in new_vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int.get(key.strip(), self.str_to_int["<|unk|>"]) for key in preprocessed]

        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!()])', r'\1', text)
        text = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', text) # remove un relevant spaces between two double qoutes
        text = re.sub(r' \' s ', "'s ", text)  # make the qoute and s of possesion (it's)
        text = re.sub(r'\'\s([^\']*?)\s\'', r"'\1'", text)  # remove un relevant spaces between two single qoutes

        return text


# text1 = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."
# text = " <|endoftext|> ".join((text1, text2))
# print(text)
#
# tokenizer = SimpleTokenizerV2(new_vocab)
# print(tokenizer.encode(text))
# print(tokenizer.decode(tokenizer.encode(text)))


import tiktoken


class MyBPETokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special=set("<|EOS|>"))

    def decode(self, ids):
        return self.tokenizer.decode(ids)


# Exercise 2.1
# text = "Akwirw ier"
# my_bpe_tokenizer = MyBPETokenizer()
# print(my_bpe_tokenizer.encode(text))
# print(my_bpe_tokenizer.decode(my_bpe_tokenizer.encode(text)))
# Done!


tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(raw_text)
# print(len(enc_text))

enc_sample = enc_text[50:]
# print(enc_sample)
