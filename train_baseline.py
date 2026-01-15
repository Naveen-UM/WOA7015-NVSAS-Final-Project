import torch
import json
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import json
import os

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

tokenizer = QuestionTokenizer(JSON_PATH)

dataset = VQARADDataset(JSON_PATH, IMAGE_DIR, answer_to_idx)

dataset = [d for d in dataset if d is not None]

loader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTM(
    vocab_size=len(tokenizer.word_to_idx),
    answer_classes=len(answer_to_idx)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

os.makedirs("results", exist_ok=True)

epoch_losses = []

epochs = 3
for epoch in range(epochs):
    total_loss = 0.0
    for images, questions, labels in loader:
        images = images.to(device)
        questions = torch.tensor(
            [tokenizer.encode(q) for q in questions],
            dtype=torch.long
        ).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, questions)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

with open("results/baseline_losses.json", "w") as f:
    json.dump({"loss": epoch_losses}, f, indent=2)

print("Baseline training completed.")