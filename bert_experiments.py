from copy import deepcopy
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, AutoModel, DataCollatorWithPadding, DistilBertForSequenceClassification
from torch import nn
from datasets import load_dataset
from collections import OrderedDict
from transformers import AdamW, get_scheduler
import torch 
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from tqdm import tqdm, trange
import evaluate as e

from load_glue import *

import argparse


max_length = 128
NUM_EPOCHS = 3
glue_type = "cola"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STUDENT_MODELS = [
    "huawei-noah/TinyBERT_General_4L_312D",
    "distilbert-base-uncased",
    "google/mobilebert-uncased",
]

teacher = "bert-base-uncased"
student = "distilbert-base-uncased"

lsm = torch.nn.LogSoftmax(dim=-1)

class DistilledModel(nn.Module):
    def __init__(self, type, task):
        super().__init__()
        num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
        self.model = DistilBertForSequenceClassification.from_pretrained(type, num_labels=num_labels)
        # print(self.model.embeddings)
    
    def forward(self, **inputs):
        x = self.model(**inputs) 
        return x

def train(model, dataloader):
    num_training_steps = NUM_EPOCHS * len(dataloader)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    print("traiing")
    model.train()
    for step in trange(NUM_EPOCHS):
        for n, inputs in enumerate(tqdm(dataloader)):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model_output_dict = model(**inputs)

            loss = model_output_dict["loss"]

            loss.backward() 
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    return model

def evaluate(model, dataloader, task):
    metric = e.load("glue", task)
    model.eval()
    
    for n, inputs in enumerate(tqdm(dataloader)):
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        if glue_type != "stsb":
            predictions = torch.argmax(logits, dim=-1)
        else:
            predictions = logits[:, 0]

        metric.add_batch(predictions=predictions, references=inputs["labels"])

    print(task, metric.compute())

    
def tokenization(tokenzier, example):
    return tokenzier(example["text"], 
            truncation=True,
            padding=True)

def main():
    parser = argparse.ArgumentParser(description="Training/Eval for GLUE datasets")
    parser.add_argument("--task", "-t", type=str, help="GLUE task")
    
    args = parser.parse_args()
    task = args.task

    assert task in GLUE_CONFIGS

    model = DistilledModel(
        "./results/s_distilbert_t_bert_data_wikitext_dataset_seed_42_mlm_True_ce_0.25_mlm_0.25_cos_0.25_causal-ce_0.25_causal-cos_0.25_nm_single_middle_layer_6_crossway_False_int-prop_0.3_consec-token_True_masked-token_False_max-int-token_-1_eff-bs_240",
        task=task)
    
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(teacher)
    train_dataset, val_dataset, _ = load_glue_dataset(tokenizer, task)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
    train_dataset = train_dataset.remove_columns(["token_type_ids"])

    val_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
    val_dataset = val_dataset.remove_columns(["token_type_ids"])
    print(train_dataset, val_dataset)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=data_collator)

    model = train(model, train_dataloader)
    evaluate(model, val_dataloader, task)

    print(task)


if __name__ == "__main__":
    main()
