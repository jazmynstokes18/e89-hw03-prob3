"""
03_dataloaders.py
Wraps the three splits from 02_load_data.py in DataLoaders. Only the
training loader is shuffled, since shuffling validation/test data doesn't
affect evaluation and just costs time.
"""

from torch.utils.data import DataLoader

BATCH_SIZE = 32

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Sanity-check what's actually coming out of the loader before building a
# model around it.
sample_images, sample_labels = next(iter(train_loader))
sample_image, sample_label = sample_images[0], sample_labels[0]

print(f"Sample image shape: {sample_image.shape}")
print(f"Sample image dtype: {sample_image.dtype}")
print(f"Sample label: {sample_label.item()} ({class_names[sample_label.item()]})")
