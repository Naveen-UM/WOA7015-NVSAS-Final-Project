import json
from collections import Counter

json_path = "data/VQA_RAD Dataset Public.json"

with open(json_path, "r") as f:
    data = json.load(f)

answers = [str(item["answer"]).strip().lower() for item in data]

counter = Counter(answers)
most_common = counter.most_common(100)

answer_to_idx = {ans: idx for idx, (ans, _) in enumerate(most_common)}

print("Top 10 answers:")
for ans, cnt in most_common[:10]:
    print(ans, cnt)

print("Total answer classes:", len(answer_to_idx))
