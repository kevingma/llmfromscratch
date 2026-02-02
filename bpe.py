from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))

from dataset import GPTDatasetV1
from torch.utils.data import DataLoader
import torch

tokenizer = tiktoken.get_encoding("gpt2")

text = (
 "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
 "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

# l o w </w>
# (l, o), (o, w), (w, <w/w>)
# merge the most frequent adjacent symbols across all words
# l ow </w
# repeat 


with open("the_verdict.txt", "r", encoding="utf-8") as f:
 raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y: {y}")
# x: [290, 4920, 2241, 287]
# y: [4920, 2241, 287, 257]

context_size = 4
for i in range(1, context_size+1):
 context = enc_sample[:i] 
 desired = enc_sample[i]
 # [290] e.g I
 # [4920] e.g am
 print(context, "---->", desired)

# [290] ----> 4920
# [290, 4920] ----> 2241
# [290, 4920, 2241] ----> 287
# [290, 4920, 2241, 287] ----> 257

for i in range(1, context_size+1):
 context = enc_sample[:i]
 desired = enc_sample[i]
 print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

#  and ---->  established
#  and established ---->  himself
#  and established himself ---->  in
#  and established himself in ---->  a

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
        )
    return dataloader

with open("the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

# [tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]

second_batch = next(data_iter)
print(second_batch)

# [tensor([[ 367, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])]

dataloader = create_dataloader_v1(
 raw_text, batch_size=8, max_length=4, stride=4,
 shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# Inputs:
#  tensor([[   40,   367,  2885,  1464],
#         [ 1807,  3619,   402,   271],
#         [10899,  2138,   257,  7026],
#         [15632,   438,  2016,   257],
#         [  922,  5891,  1576,   438],
#         [  568,   340,   373,   645],
#         [ 1049,  5975,   284,   502],
#         [  284,  3285,   326,    11]])

# Targets:
#  tensor([[  367,  2885,  1464,  1807],
#         [ 3619,   402,   271, 10899],
#         [ 2138,   257,  7026, 15632],
#         [  438,  2016,   257,   922],
#         [ 5891,  1576,   438,   568],
#         [  340,   373,   645,  1049],
#         [ 5975,   284,   502,   284],
#         [ 3285,   326,    11,   287]])

next_inputs, next_targets = next(data_iter)
print("Inputs:\n", next_inputs)
print("\nTargets:\n", next_targets)

# Inputs:
#  tensor([[  287,   262,  6001,   286],
#         [  465, 13476,    11,   339],
#         [  550,  5710,   465, 12036],
#         [   11,  6405,   257,  5527],
#         [27075,    11,   290,  4920],
#         [ 2241,   287,   257,  4489],
#         [   64,   319,   262, 34686],
#         [41976,    13,   357, 10915]])

# Targets:
#  tensor([[  262,  6001,   286,   465],
#         [13476,    11,   339,   550],
#         [ 5710,   465, 12036,    11],
#         [ 6405,   257,  5527, 27075],
#         [   11,   290,  4920,  2241],
#         [  287,   257,  4489,    64],
#         [  319,   262, 34686, 41976],
#         [   13,   357, 10915,   314]])

# stride=4, max_length=4
# S0 = tokens[0:4]
# S1 = tokens[4:8]
# S2 = tokens[8:12]
# S3 = tokens[12:16]
# S4 = tokens[16:20]\


vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Createe list of lists of token IDs
max_length = 4
dataloader = create_dataloader_v1(
 raw_text, batch_size=8, max_length=max_length,
 stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# Token IDs:
#  tensor([[   40,   367,  2885,  1464],
#         [ 1807,  3619,   402,   271],
#         [10899,  2138,   257,  7026],
#         [15632,   438,  2016,   257],
#         [  922,  5891,  1576,   438],
#         [  568,   340,   373,   645],
#         [ 1049,  5975,   284,   502],
#         [  284,  3285,   326,    11]])
# Inputs shape:
#  torch.Size([8, 4])

# Creates weights for each token ID
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# torch.Size([8, 4, 256])

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)