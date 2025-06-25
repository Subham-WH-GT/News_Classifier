import torch
import numpy as np
from transformers import AutoModel, BertTokenizerFast
import torch.nn as nn

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model base
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert = AutoModel.from_pretrained('bert-base-uncased')

# Define the architecture used during training
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Load the fine-tuned model weights
model = BERT_Arch(bert)
model.load_state_dict(torch.load("bert_fakenews_best.pt", map_location=device))
model.to(device)
model.eval()

# Unseen news samples
unseen_news_text = [
    "Israel-Iran War LIVE:  Iran and Israel on Tuesday agreed to accept US President Donald Trump's ceasefire proposal to end their 12-day war that roiled the Middle East. The acceptance of the deal came after both the Jewish state and the Islamic republic final onslaught of missiles at each other. The Israeli strikes killed nine Iranians, including a nuclear scientist, the Iranian state media said.",     # Fake
    "BREAKING: Govt to Deposit ₹5 Lakh in Every Indian’s Account Under New Scheme Starting July 1st – Check Eligibility Now!"
]

# Tokenize and encode
MAX_LENGTH = 200  # Use same as training
tokens_unseen = tokenizer.batch_encode_plus(
    unseen_news_text,
    max_length=MAX_LENGTH,
    padding='max_length',
    truncation=True,
    return_tensors="pt"
)

unseen_seq = tokens_unseen['input_ids'].to(device)
unseen_mask = tokens_unseen['attention_mask'].to(device)

# Predict
with torch.no_grad():
    preds = model(unseen_seq, unseen_mask)
    preds = preds.detach().cpu().numpy()

# Get labels
labels = np.argmax(preds, axis=1)
label_map = {0: "Real", 1: "Fake"}

# Print predictions
print("\n Unseen News Predictions:")
for news, pred in zip(unseen_news_text, labels):
    print(f"- {label_map[pred]} ➤ {news}")
