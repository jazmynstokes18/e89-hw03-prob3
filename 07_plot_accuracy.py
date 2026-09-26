"""
07_plot_accuracy.py
Plots training vs. validation accuracy on one figure, plus a second figure
for training loss. Training accuracy for epoch i is an average of batch
accuracies computed *while the weights were still changing* across that
epoch, so it's plotted at the epoch midpoint (i - 0.5) rather than the
epoch's end -- validation accuracy, computed with the epoch's final
weights, is plotted at the epoch's end (i).
"""

n_epochs = len(history["train_metrics"])
epochs = list(range(1, n_epochs + 1))
train_x = [e - 0.5 for e in epochs]  # epoch midpoints

# --- Figure 1: accuracy ---
plt.figure()
plt.plot(train_x, history["train_metrics"], label="Training accuracy", marker="o")
plt.plot(epochs, history["val_metrics"], label="Validation accuracy", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs. Validation Accuracy")
plt.ylim(0, 1)
plt.legend()
plt.show()

print(f"Final training accuracy:   {history['train_metrics'][-1]:.4f}")
print(f"Final validation accuracy: {history['val_metrics'][-1]:.4f}")

# --- Figure 2: training loss ---
plt.figure()
plt.plot(epochs, history["train_losses"], label="Training loss", color="tab:red", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.show()
