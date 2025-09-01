import torch
from torchvista import trace_model
from cifar_single_gpu import Simple_CNN

model = Simple_CNN()
example_input = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image

trace_model(model, example_input)