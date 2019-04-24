import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

n = 50
rs = torch.tensor([1.] * n, requires_grad=True)
dt = 0.1

print(rs)