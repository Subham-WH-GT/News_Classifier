
# import pandas as pd

# file_path = "news.csv"
# df = pd.read_csv(file_path)





# # # Step 3: Clean the dataset (remove rows with missing labels or texts if needed)
# # df = df.dropna(subset=['text', 'label'])

# # # Step 4: Count the number of samples for each class
# # label_counts = df['label'].value_counts()
# # print("Label Counts:")
# # print(label_counts)

# # # Step 5: Calculate the percentage distribution of each class
# # label_percentages = df['label'].value_counts(normalize=True) * 100
# # print("\nLabel Percentages:")
# # print(label_percentages)

# # # Optional: Check if roughly balanced
# # threshold = 10  # max acceptable % difference
# # diff = abs(label_percentages[0] - label_percentages[1])
# # print("\nBalanced:" if diff <= threshold else "\nImbalanced:", f"Class difference = {diff:.2f}%")


# # drop the records having no text/news body
# df = df.dropna(subset=["text"]) 
# print(df.isnull().sum()) 

# # combine title and text in one feature
# df["content"] = df["title"] + " " + df["text"] 

# # make the dataset ready
# df = df[["content", "label"]]






# # FINE_TUNING 


# from sklearn.model_selection import train_test_split

# train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
# val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)


# # from transformers import AutoTokenizer
# from transformers import DebertaV2Tokenizer

# tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")

# tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
# # tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
# # tokenizer = AutoTokenizer.from_pretrained(
# #     "microsoft/deberta-v3-base",
# #     use_fast=False  # 👈 forces the use of the slower but compatible tokenizer
# # )

# # def tokenize(batch):
# #     return tokenizer(batch['content'], padding="max_length", truncation=True, max_length=512)





# def tokenize(batch):
#     return tokenizer(
#         list(batch["content"]),
#         padding="max_length",
#         truncation=True,
#         max_length=512
#     )

# train_dataset = train_dataset.map(tokenize, batched=True)





# #CONVERT TO HUGGING FACE DATASET

# from datasets import Dataset

# train_dataset = Dataset.from_pandas(train_df)
# val_dataset = Dataset.from_pandas(val_df)
# test_dataset = Dataset.from_pandas(test_df)

# train_dataset = train_dataset.map(tokenize, batched=True)
# val_dataset = val_dataset.map(tokenize, batched=True)
# test_dataset = test_dataset.map(tokenize, batched=True)

# train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
# val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
# test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])


# # SET THE HYPER_PARAMETERS 

# from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=2)

# training_args = TrainingArguments(

#     output_dir="./deberta-fake-news",
#     per_device_train_batch_size=16,       
#     per_device_eval_batch_size=16,        
#     num_train_epochs=4,                   
#     evaluation_strategy="epoch",         
#     save_strategy="epoch",               
#     logging_strategy="steps",            
#     logging_steps=50,                    
#     learning_rate=2e-5,                 
#     weight_decay=0.01,                   
#     save_total_limit=2,                  
#     load_best_model_at_end=True,         
#     metric_for_best_model="accuracy",    
#     fp16=True,                           
#     seed=42,                             
#     report_to="none"

# )

# import evaluate

# accuracy = evaluate.load("accuracy")
# f1 = evaluate.load("f1")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = logits.argmax(axis=1)
#     return {
#         "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
#         "f1": f1.compute(predictions=preds, references=labels)["f1"]
#     }


# # TRAIN_MODEL

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     compute_metrics=compute_metrics
# )

# trainer.train()


# # TEST_MODEL

# trainer.evaluate(test_dataset)











import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import evaluate

# STEP 1: Load and clean dataset
file_path = "news.csv"
df = pd.read_csv(file_path)

df = df.dropna(subset=["text"]) 
df["title"] = df["title"].fillna("")  # fill empty titles with blank
df["content"] = df["title"] + " " + df["text"]
df = df[["content", "label"]]

# STEP 2: Split dataset
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

# STEP 3: Load tokenizer
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")

def tokenize(batch):
    return tokenizer(
        list(batch["content"]),
        padding="max_length",
        truncation=True,
        max_length=512
    )

# STEP 4: Convert to Hugging Face Dataset & tokenize
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# STEP 5: Load model
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=2)

# STEP 6: Training arguments
training_args = TrainingArguments(
    output_dir="./deberta-fake-news",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    seed=42,
    report_to="none", 
    
)

# STEP 7: Metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels)["f1"]
    }

# STEP 8: Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# STEP 9: Train model
trainer.train()

# STEP 10: Evaluate on test set
trainer.evaluate(test_dataset)
