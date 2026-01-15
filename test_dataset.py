from dataset import VQARADDataset

dataset = VQARADDataset(
    json_path="data/VQA_RAD Dataset Public.json",
    image_dir="data/images"
)

print("Total samples:", len(dataset))

image, question, answer = dataset[0]

print("Image shape:", image.shape)
print("Question:", question)
print("Answer:", answer)
