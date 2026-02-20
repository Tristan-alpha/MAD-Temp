import torch
torch.set_default_device('cuda') 

B = 4096
D = 4096
a = torch.linspace(-1000, 1000, B*D).reshape(B, D)
print(a.shape)
b = torch.linspace(-1000, 1000, D*D).reshape(D, D)
print(b.shape)
# Doing a matrix vector multiplication by taking
# the first element of the batch
out1 = torch.mm(a[:1], b)
# Doing a matrix matrix multiplication and then taking
# the first element of the batch
out2 = torch.mm(a, b)[:1]
print((out1 - out2).abs().max()) # tensor(1669.2500, device='cuda:0')