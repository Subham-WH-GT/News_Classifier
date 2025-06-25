

import json
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import requests
from transformers import (
    DebertaV2Tokenizer, 
    AutoModelForSequenceClassification, 
    BertTokenizerFast, 
    AutoModel
)
from newspaper import Article
from openai import OpenAI
from duckduckgo_search import DDGS 
from dotenv import load_dotenv
import os

load_dotenv()
# --- Config --- 


MODEL_PATH = os.getenv("MODEL_PATH")

BERT_MODEL_PATH = os.getenv("BERT_MODEL_PATH")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")

# --- DeBERTa Setup ---
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
label_map = {0: "Fake", 1: "Real"}

# --- OpenAI LLM Client ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- BERT Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert_base = AutoModel.from_pretrained('bert-base-uncased')

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

bert_model = BERT_Arch(bert_base)
bert_model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=device))
bert_model.to(device)
bert_model.eval()

# --- Helper Functions ---

def web_search(query, max_results=3):
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region="wt-wt", safesearch="strict", max_results=max_results)
            return list(results)
    except Exception as e:
        print(f" DuckDuckGo Error: {e}")
        return []

def extract_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text[:1500]
    except:
        return ""

def classify_deberta(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_class].item()
    return label_map[predicted_class], round(confidence, 4)

def classify_bert(text):
    tokens = bert_tokenizer.batch_encode_plus(
        [text],
        max_length=200,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    input_ids = tokens['input_ids'].to(device)
    mask = tokens['attention_mask'].to(device)

    with torch.no_grad():
        output = bert_model(input_ids, mask)
        pred = torch.argmax(output, dim=1).item()
    return "Fake" if pred == 1 else "Real"

def classify_llm(post):
    query = post['title']
    results = web_search(query)

    sources_text = ""
    for res in results:
        content = extract_article_content(res['href'])
        if content:
            sources_text += f"\n\nSource: {res['title']}\nURL: {res['href']}\nContent:\n{content}"

    top_comments = "\n".join(post.get("top_comments", []))
    prompt = f"""
You are a fake news detection expert. A Reddit post has made the following claim:

Title: {post['title']}
URL: {post['url']}
Reddit Comments: {top_comments}

You have also found the following content from trusted news sources via web search:
{sources_text}

Based on the above, is this news Real or Fake? Reply only "Real" or "Fake" and give a short justification (1-2 lines).
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()
        if "Fake" in content:
            return "Fake", content
        elif "Real" in content:
            return "Real", content
        else:
            return "Unknown", content
    except Exception as e:
        return "Error", str(e)

# --- Load Reddit Posts ---
with open("reddit_posts.json", "r", encoding="utf-8") as f:
    reddit_posts = json.load(f)

# --- Main Classification Loop ---
for post in reddit_posts:
    title = post.get("title", "")
    top_comments = "\n".join(post.get("top_comments", []))
    input_text = f"{title}\n\n{top_comments}".strip()

    deberta_label, deberta_conf = classify_deberta(input_text)
    llm_label, llm_reason = classify_llm(post)

    # Decision Logic
    if deberta_label == llm_label:
        final_label = llm_label
        reason = f"Both agreed. LLM justification: {llm_reason}"
    else:
        bert_label = classify_bert(input_text)
        final_label = bert_label
        reason = (
            f"Disagreement ➤ DeBERTa: {deberta_label}, LLM: {llm_label}\n"
            f"Used BERT fallback ➤ Final: {bert_label}"
        )

    print(f"\n Title: {title}")
    print(f"Final Verdict: {final_label}")
    print(f" Reason: {reason}\n")

    time.sleep(10) 



