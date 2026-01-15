import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

models = ["CNN–LSTM Baseline", "CLIP"]
accuracies = [0.26, 25.67]

plt.figure()
plt.bar(models, accuracies)
plt.ylabel("Accuracy (%)")
plt.title("Overall Accuracy Comparison on VQA-RAD")
plt.grid(axis="y")

plt.savefig("results/overall_model_comparison.png", dpi=200, bbox_inches="tight")
plt.show()

question_types = ["CLOSED", "OPEN"]
clip_accuracies = [43.73, 0.95]

plt.figure()
plt.bar(question_types, clip_accuracies)
plt.ylabel("Accuracy (%)")
plt.title("CLIP Performance on CLOSED vs OPEN Questions")
plt.grid(axis="y")

plt.savefig("results/clip_closed_open_comparison.png", dpi=200, bbox_inches="tight")
plt.show()
