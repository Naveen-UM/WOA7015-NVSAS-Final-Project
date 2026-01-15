import json
import matplotlib.pyplot as plt

with open("results/clip_running_accuracy.json", "r") as f:
    data = json.load(f)

x = data["x"]
overall = data["overall"]
closed = data["closed"]
open_ = data["open"]

plt.figure()
plt.plot(x, overall, marker="o", label="Overall")
plt.plot(x, closed, marker="o", label="CLOSED")
plt.plot(x, open_, marker="o", label="OPEN")

plt.xlabel("Number of evaluated samples")
plt.ylabel("Accuracy")
plt.title("CLIP Running Accuracy on VQA-RAD")
plt.legend()
plt.grid(True)

plt.savefig("results/clip_running_accuracy.png", dpi=200, bbox_inches="tight")
plt.show()
