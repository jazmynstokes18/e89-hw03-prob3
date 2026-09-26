"""
02_load_data.py
Loads Fashion MNIST via torchvision, converts images to float tensors scaled
to [0, 1] with transforms.v2, and splits the 60k training images into a
55k train / 5k validation split. Depends on: torch, torchvision (imported
here since 01_setup.py doesn't need it), device/seed from 01_setup.py.
"""

from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import random_split

DATA_DIR = "datasets"

# Convert PIL images -> float32 tensors scaled to [0, 1] (no normalization,
# just the raw [0,1] range as requested).
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

# Full 60k-image training set and the 10k-image test set.
train_data_full = datasets.FashionMNIST(
    root=DATA_DIR, train=True, download=True, transform=transform
)
test_data = datasets.FashionMNIST(
    root=DATA_DIR, train=False, download=True, transform=transform
)

class_names = train_data_full.classes

# Re-seed right before the random split so the 55k/5k split is reproducible
# on its own, independent of anything that happened above.
torch.manual_seed(SEED)
train_data, val_data = random_split(train_data_full, [55000, 5000])

print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(val_data)}")
print(f"Test size: {len(test_data)}")
print(f"Class names: {class_names}")
