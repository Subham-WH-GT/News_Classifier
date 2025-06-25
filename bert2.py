from transformers import BertTokenizerFast, BertForSequenceClassification

# Load tokenizer and base model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Load your fine-tuned weights
import torch
state_dict = torch.load("bert_fakenews_best.pt", map_location="cpu")
model.load_state_dict(state_dict)

# Save model and tokenizer to a folder
model.save_pretrained("bert-fake-news-hf")
tokenizer.save_pretrained("bert-fake-news-hf")
