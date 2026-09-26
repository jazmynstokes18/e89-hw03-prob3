"""
08_predict.py
Runs the trained model on 3 validation images: prints predicted vs. actual
class names, then softmaxes the logits and shows the top-4 predicted
probabilities per image. round() isn't implemented on MPS, so tensors are
moved to cpu before rounding. Finally shows the 3 images with their labels.
"""

N_SAMPLES = 3

model.eval()
images, labels = next(iter(val_loader))
images, labels = images[:N_SAMPLES], labels[:N_SAMPLES]

with torch.no_grad():
    logits = model(images.to(device))

predicted = logits.argmax(dim=1)

# round() isn't implemented on MPS -- move to cpu first.
probs = F.softmax(logits, dim=1).cpu()
predicted_cpu = predicted.cpu()

print("Predictions:")
for i in range(N_SAMPLES):
    actual_name = class_names[labels[i].item()]
    predicted_name = class_names[predicted_cpu[i].item()]
    print(f"  Image {i}: predicted = {predicted_name!r}, actual = {actual_name!r}")

print("\nTop-4 predicted probabilities per image:")
top4_probs, top4_idx = torch.topk(probs, 4, dim=1)
for i in range(N_SAMPLES):
    print(f"  Image {i}:")
    for prob, idx in zip(top4_probs[i], top4_idx[i]):
        print(f"    {class_names[idx.item()]:<12s} {round(prob.item(), 4)}")

# --- Show the images with predicted vs. actual labels ---
fig, axes = plt.subplots(1, N_SAMPLES, figsize=(9, 3))
for i, ax in enumerate(axes):
    ax.imshow(images[i].squeeze(), cmap="gray")
    actual_name = class_names[labels[i].item()]
    predicted_name = class_names[predicted_cpu[i].item()]
    ax.set_title(f"Pred: {predicted_name}\nActual: {actual_name}", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()
