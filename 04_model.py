"""
04_model.py
Defines the MLP classifier: Flatten -> Linear(784,300) -> ReLU ->
Linear(300,100) -> ReLU -> Linear(100,10). No activation on the final layer
because nn.CrossEntropyLoss applies log-softmax internally and expects raw
logits.
"""


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28 * 28, 300)
        self.linear2 = nn.Linear(300, 100)
        self.linear3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)  # raw logits, no activation
        return x


model = ImageClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()

num_params = sum(p.numel() for p in model.parameters())
print(f"Model: {model}")
print(f"Total trainable parameters: {num_params:,}")
