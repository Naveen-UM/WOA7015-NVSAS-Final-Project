import json
from collections import Counter

class QuestionTokenizer:
    def __init__(self, json_path, max_length=20):
        with open(json_path, "r") as f:
            data = json.load(f)

        questions = [item["question"].lower().split() for item in data]
        counter = Counter(word for q in questions for word in q)

        self.word_to_idx = {
            "<PAD>": 0,
            "<UNK>": 1
        }

        for word in counter:
            self.word_to_idx[word] = len(self.word_to_idx)

        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        self.max_length = max_length

    def encode(self, question):
        tokens = question.lower().split()
        encoded = [
            self.word_to_idx.get(tok, self.word_to_idx["<UNK>"])
            for tok in tokens
        ]

        if len(encoded) < self.max_length:
            encoded += [0] * (self.max_length - len(encoded))
        else:
            encoded = encoded[:self.max_length]

        return encoded
