
import numpy as np
import pandas as pd
from transformers import AutoModel, BertTokenizerFast
import torch
import os
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set path
os.chdir('C:/Users/hivyr/Desktop/News/')

# Load data
true_data = pd.read_csv('True.csv')
fake_data = pd.read_csv('Fake.csv')

true_data['Target'] = ['True'] * len(true_data)
fake_data['Target'] = ['Fake'] * len(fake_data)

data = pd.concat([true_data, fake_data], ignore_index=True).sample(frac=1).reset_index(drop=True)
data['label'] = pd.get_dummies(data.Target)['Fake']

# Split data
train_text, temp_text, train_labels, temp_labels = train_test_split(
    data['text'], data['label'], random_state=2018, test_size=0.3, stratify=data['Target']
)

val_text, test_text, val_labels, test_labels = train_test_split(
    temp_text, temp_labels, random_state=2018, test_size=0.5, stratify=temp_labels
)

# Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

MAX_LENGTH = 200
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length=MAX_LENGTH,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

test_seq = tokens_test['input_ids']
test_mask = tokens_test['attention_mask']
test_y = torch.tensor(test_labels.tolist())

# Define BERT architecture
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

# Load BERT base model and wrap it
bert = AutoModel.from_pretrained('bert-base-uncased')
model = BERT_Arch(bert)
model.load_state_dict(torch.load('bert_fakenews_best.pt', map_location=device))
model.to(device)
model.eval()

# Run test in batches
test_data = TensorDataset(test_seq, test_mask)
test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=32)

all_preds = []

with torch.no_grad():
    for batch in test_dataloader:
        sent_id, mask = [t.to(device) for t in batch]
        output = model(sent_id, mask)
        all_preds.append(output.cpu())

# Convert predictions to numpy and compute final output
preds = torch.cat(all_preds, dim=0).numpy()
preds = np.argmax(preds, axis=1)

print("Classification Report:\n")
print(classification_report(test_y, preds))
