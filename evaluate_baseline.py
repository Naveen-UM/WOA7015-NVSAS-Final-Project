import torch
import json
from torch.utils.data import DataLoader
from dataset import VQARADDataset
from tokenizer import QuestionTokenizer
from baseline_model import CNNLSTM

JSON_PATH = "data/VQA_RAD Dataset Public.json"
IMAGE_DIR = "data/images"

with open(JSON_PATH, "r") as f:
    raw_data = json.load(f)

answers = [str(item["answer"]).strip().lower() for item in raw_data]
unique_answers = list(dict.fromkeys(answers))[:100]
answer_to_idx = {a: i for i, a in enumerate(unique_answers)}
idx_to_answer = {i: a for a, i in answer_to_idx.items()}

tokenizer = QuestionTokenizer(JSON_PATH)

dataset = VQARADDataset(JSON_PATH, IMAGE_DIR, answer_to_idx)
dataset = [d for d in dataset if d is not None]

loader = DataLoader(dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTM(
    vocab_size=len(tokenizer.word_to_idx),
    answer_classes=len(answer_to_idx)
).to(device)

model.eval()

correct = 0
total = 0
examples = []

with torch.no_grad():
    for images, questions, labels in loader:
        images = images.to(device)
        questions = torch.tensor(
            [tokenizer.encode(q) for q in questions],
            dtype=torch.long
        ).to(device)
        labels = labels.to(device)

        outputs = model(images, questions)
        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if len(examples) < 10:
            for i in range(min(2, labels.size(0))):
                examples.append({
                    "question": questions[i].cpu().numpy().tolist(),
                    "predicted": idx_to_answer[preds[i].item()],
                    "ground_truth": idx_to_answer[labels[i].item()]
                })

accuracy = correct / total
print(f"Baseline Accuracy: {accuracy:.4f}")

print("\nSample Predictions:")
for ex in examples:
    print(ex)
