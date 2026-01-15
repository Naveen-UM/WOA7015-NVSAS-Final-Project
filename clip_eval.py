import json
import os
from PIL import Image
from collections import Counter

import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

JSON_PATH = "data/VQA_RAD Dataset Public.json"
IMAGE_DIR = "data/images"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


CLIP_NAME = "openai/clip-vit-base-patch32"
TOPK_OPEN = 50         
LOG_EVERY = 50        

def norm_text(x):
    return str(x).strip().lower()


def build_top_answers(data, k=100):
    answers = [norm_text(item["answer"]) for item in data]
    counter = Counter(answers)
    return [ans for ans, _ in counter.most_common(k)]


def get_candidates(item, top_open):
    a_type = str(item.get("answer_type", "")).strip().upper()
    if a_type == "CLOSED":
        return ["yes", "no"]
    return top_open


def predict_clip(model, processor, device, image, question, candidates):
    img_inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        img_feat = model.get_image_features(**img_inputs)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    texts = [f"question: {question} answer: {c}" for c in candidates]
    txt_inputs = processor(
        text=texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    with torch.no_grad():
        txt_feat = model.get_text_features(**txt_inputs)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    sims = (img_feat @ txt_feat.T).squeeze(0)
    best_idx = int(torch.argmax(sims).item())
    return candidates[best_idx]

def main():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    top_answers = build_top_answers(data, k=100)
    top_open = top_answers[:TOPK_OPEN]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(CLIP_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_NAME)

    total = correct = 0
    closed_total = closed_correct = 0
    open_total = open_correct = 0

    xs = []
    overall_acc = []
    closed_acc = []
    open_acc = []

    examples = []

    for item in tqdm(data, desc="Evaluating CLIP"):
        img_path = os.path.join(IMAGE_DIR, item["image_name"])
        if not os.path.exists(img_path):
            continue

        question = item["question"]
        gt = norm_text(item["answer"])
        a_type = str(item.get("answer_type", "")).strip().upper()

        image = Image.open(img_path).convert("RGB")
        candidates = get_candidates(item, top_open)

        pred = predict_clip(model, processor, device, image, question, candidates)

        total += 1
        is_correct = (pred == gt)
        if is_correct:
            correct += 1

        if a_type == "CLOSED":
            closed_total += 1
            if is_correct:
                closed_correct += 1
        else:
            open_total += 1
            if is_correct:
                open_correct += 1

        if len(examples) < 10:
            examples.append({
                "answer_type": a_type,
                "question": question,
                "ground_truth": gt,
                "predicted": pred
            })

        if total % LOG_EVERY == 0:
            xs.append(total)
            overall_acc.append(correct / total)
            closed_acc.append(closed_correct / closed_total if closed_total > 0 else 0)
            open_acc.append(open_correct / open_total if open_total > 0 else 0)

    with open(f"{RESULTS_DIR}/clip_running_accuracy.json", "w") as f:
        json.dump({
            "x": xs,
            "overall": overall_acc,
            "closed": closed_acc,
            "open": open_acc
        }, f, indent=2)

    print("\n=== FINAL CLIP RESULTS ===")
    print(f"Overall: {correct/total:.4f}")
    print(f"CLOSED : {closed_correct/closed_total:.4f}")
    print(f"OPEN   : {open_correct/open_total:.4f}")

    print("\nSample Predictions:")
    for ex in examples:
        print(ex)


if __name__ == "__main__":
    main()
