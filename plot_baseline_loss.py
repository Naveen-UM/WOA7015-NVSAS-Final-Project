import json
import matplotlib.pyplot as plt

with open("results/baseline_losses.json", "r") as f:
    data = json.load(f)

losses = data["loss"]
epochs = list(range(1, len(losses) + 1))

plt.figure()
plt.plot(epochs, losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Baseline CNN+LSTM Training Loss")
plt.xticks(epochs)
plt.grid(True)

plt.savefig("results/baseline_loss_curve.png", dpi=200, bbox_inches="tight")
plt.show()
