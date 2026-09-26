"""
06_run_training.py
Actually trains the model built in 04_model.py using the functions defined
in 05_train.py: 20 epochs, plain SGD with lr=0.1, and a torchmetrics
multiclass accuracy metric living on the same device as the model.
"""

N_EPOCHS = 20
LEARNING_RATE = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
metric = torchmetrics.classification.MulticlassAccuracy(num_classes=10).to(device)

history = train2(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    metric=metric,
    device=device,
    n_epochs=N_EPOCHS,
)
