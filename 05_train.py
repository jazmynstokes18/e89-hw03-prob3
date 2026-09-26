"""
05_train.py
Just function definitions (evaluate() and train2()) -- nothing is run here.
06_run_training.py calls these on the model built in 04_model.py.
"""


def evaluate(model, loader, metric, device):
    """Run one full pass over `loader` under no_grad and return
    (avg_loss, metric_value)."""
    model.eval()
    metric.reset()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            n_batches += 1
            metric.update(logits, y)
    avg_loss = total_loss / n_batches
    metric_value = metric.compute().item()
    return avg_loss, metric_value


def train2(model, train_loader, val_loader, optimizer, loss_fn, metric, device, n_epochs):
    """Train for n_epochs, printing metrics each epoch, and return a history
    dict with train_losses, train_metrics, val_losses, val_metrics -- so it
    can be plotted later instead of just printed."""
    history = {
        "train_losses": [],
        "train_metrics": [],
        "val_losses": [],
        "val_metrics": [],
    }

    for epoch in range(n_epochs):
        model.train()
        metric.reset()
        running_loss = 0.0
        n_batches = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            metric.update(logits, y)

        train_loss = running_loss / n_batches
        train_metric = metric.compute().item()

        val_loss, val_metric = evaluate(model, val_loader, metric, device)

        history["train_losses"].append(train_loss)
        history["train_metrics"].append(train_metric)
        history["val_losses"].append(val_loss)
        history["val_metrics"].append(val_metric)

        print(
            f"Epoch {epoch + 1}/{n_epochs} - "
            f"train_loss: {train_loss:.4f}, train_acc: {train_metric:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_metric:.4f}"
        )

    return history
