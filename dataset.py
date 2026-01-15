import json
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class VQARADDataset(Dataset):
    def __init__(self, json_path, image_dir, answer_to_idx):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.image_dir = image_dir
        self.answer_to_idx = answer_to_idx

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_name = item["image_name"]
        question = item["question"]
        answer = str(item["answer"]).strip().lower()

        if answer not in self.answer_to_idx:
            return None

        label = self.answer_to_idx[answer]

        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, question, label
