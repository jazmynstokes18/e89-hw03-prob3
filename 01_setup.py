"""
01_setup.py
HW03 Problem 3 - Fashion MNIST classifier

All nine scripts in this repo run in order and share ONE namespace (e.g. via
`exec(open(f).read())` in a single Python session, or by running them as
consecutive cells in a notebook). Imports and setup done here are reused by
every later script, so nothing below is re-imported downstream.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import matplotlib.pyplot as plt

# --- Pick the best available device: cuda > mps > cpu ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# --- Seed everything for reproducibility ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Sensible matplotlib defaults used by every plotting script later on ---
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["font.size"] = 11

print(f"PyTorch version: {torch.__version__}")
print(f"Using device: {device}")
