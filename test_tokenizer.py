from tokenizer import QuestionTokenizer

tokenizer = QuestionTokenizer(
    json_path="data/VQA_RAD Dataset Public.json"
)

question = "Are regions of the brain infarcted?"
encoded = tokenizer.encode(question)

print("Encoded question:", encoded)
print("Length:", len(encoded))
