import torch

input_ids = torch.tensor([2, 3, 5, 1])

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

# Parameter containing:
# tensor([[ 0.3374, -0.1778, -0.1690],
#         [ 0.9178,  1.5810,  1.3010],
#         [ 1.2753, -0.2010, -0.1606],
#         [-0.4015,  0.9666, -1.1481],
#         [-1.1589,  0.3255, -0.6315],
#         [-2.8400, -0.7849, -1.4096]], requires_grad=True)


print(embedding_layer(torch.tensor([3])))

# tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)

# [1, 0, 0, 0, 0, 0]
# [0, 1, 0, 0, 0, 0]

print(embedding_layer(input_ids))