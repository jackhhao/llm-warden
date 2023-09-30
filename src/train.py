from datasets import load_dataset, ClassLabel
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
import evaluate

labels = ClassLabel(names=["benign", "jailbreak"])

# prepare and tokenize dataset
dataset = (load_dataset("csv", data_dir = "../datasets/balanced", data_files={"train": "jailbreak_dataset_train_balanced.csv", "test": "jailbreak_dataset_test_balanced.csv"})
           .rename_column("prompt", "text")
           .rename_column("type", "label"))
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# map labels to IDs
id2label = {0: "benign", 1: "jailbreak"}
label2id = {value: key for key,value in id2label.items()}

def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenized['label'] = labels.str2int(examples['label'])
    return tokenized

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(200))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(200))

# set up evaluation 
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# load pretrained model and evaluate model after each epoch
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)
training_args = TrainingArguments(
    output_dir="../training/",
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("../model/")