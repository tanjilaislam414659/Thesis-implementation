import sys
import networkx as nx
import torch

print("Python version:", sys.version)
print("NetworkX version:", nx.__version__)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())