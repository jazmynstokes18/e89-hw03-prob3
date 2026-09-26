"""
09_evaluate.py
Final, one-time run on the held-out test set. Reuses evaluate() from
05_train.py and the metric from 06_run_training.py.
"""

test_loss, test_accuracy = evaluate(model, test_loader, metric, device)

print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Total trainable parameters: {num_params:,}")
