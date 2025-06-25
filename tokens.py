from transformers import DebertaV2Tokenizer

# Load tokenizer from base model
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")

# Save it to the same directory where your fine-tuned model is saved
tokenizer.save_pretrained("./deberta-fake-news")

print("Tokenizer saved successfully.")
