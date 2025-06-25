# from transformers import AutoConfig

# # Load config from base model
# config = AutoConfig.from_pretrained("microsoft/deberta-v3-base")

# # Save it to your fine-tuned model directory
# config.save_pretrained("C:\\Users\\hivyr\\Desktop\\News\\deberta-fake-news")

# print("Model config saved successfully.")





from transformers import AutoModelForSequenceClassification

# Path to your latest/best checkpoint
checkpoint_path = "C:\\Users\\hivyr\\Desktop\\News\\deberta-fake-news\\checkpoint-14420"
save_path = "C:\\Users\\hivyr\\Desktop\\News\\deberta-fake-news"

# Load the model from the checkpoint (supports safetensors automatically)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

# Save to top-level folder
model.save_pretrained(save_path)

print("✅ Model successfully saved to:", save_path)


