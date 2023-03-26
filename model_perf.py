from copy import deepcopy
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, AutoModel, DataCollatorWithPadding, DistilBertForSequenceClassification
import numpy as np
from torch import nn
from datasets import load_dataset
from collections import OrderedDict, defaultdict
from transformers import AdamW, get_scheduler
import torch 
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from tqdm import tqdm, trange
import evaluate as e
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac

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
    print("training")
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
    
    
def tokenization(tokenzier, example):
    return tokenzier(example["text"], 
            truncation=True,
            padding=True)
   
   
def output_to_predictions(outputs, glue_type):
    '''helper function that transform model output to predictions'''
    logits = outputs.logits
    if glue_type != "stsb":
        predictions = torch.argmax(logits, dim=-1)
    else:
        predictions = logits[:, 0]
    return predictions


def perturb_inputs(inputs, tokenizer, augmenter):
    '''helper function to perturb inputs, so as to evaluate model robustness'''
    augmented_inputs = inputs.copy()
    augmented_inputs["input_ids"], augmented_inputs["attention_mask"] = [], []
    for i in range(len(inputs["input_ids"])):
        text = tokenizer.decode(inputs["input_ids"][i], skip_special_tokens=True)
        augmented_text = augmenter.augment(text)
        augmented_input = tokenizer(augmented_text, max_length=len(inputs["input_ids"][i]),
                                    padding='max_length', truncation=True)
        augmented_inputs["input_ids"] += augmented_input["input_ids"]
        augmented_inputs["attention_mask"] += augmented_input["attention_mask"]
    augmented_inputs["input_ids"] = torch.tensor(augmented_inputs["input_ids"]).to(device)
    augmented_inputs["attention_mask"] = torch.tensor(augmented_inputs["attention_mask"]).to(device)
    return augmented_inputs


def evaluate(model, dataloader, tokenizer, task):
    '''
    evaluate model performance in terms of metrics,
    as well as robustness as model performance when input gets perturbed
    '''
    model.eval()
    # original model performance
    metric = e.load("glue", task)
    for n, inputs in enumerate(tqdm(dataloader)):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = output_to_predictions(outputs, glue_type)
        metric.add_batch(predictions=predictions, references=inputs["labels"])
    original_scores = metric.compute()

    # robustness - replace characters
    charsub_scores = defaultdict(list)
    for _ in range(10):
        metric2 = e.load("glue", task)
        aug_charsub = nac.RandomCharAug(action='substitute', aug_char_min=1, aug_char_max=1, aug_word_p=0.1,
                                        candidates=list('abcdefghijklmnopqrstuvwxyz'))
        for n, inputs in enumerate(tqdm(dataloader)):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # perturb inputs
            augmented_inputs = perturb_inputs(inputs, tokenizer, aug_charsub)
            # get predictions
            with torch.no_grad():
                outputs = model(**augmented_inputs)
            predictions = output_to_predictions(outputs, glue_type)
            # add evaluation sub-results
            metric2.add_batch(predictions=predictions, references=augmented_inputs["labels"])
        # aggregate evaluation results
        charsub_scores_i = metric2.compute()
        for key, val in charsub_scores_i.items():
            charsub_scores[key].append(val)

    # robustness - swap adjacent word
    wordswap_scores = defaultdict(list)
    for _ in range(10):
        metric3 = e.load("glue", task)
        aug_wordswap = naw.RandomWordAug(action="swap", aug_p=0.2)
        for n, inputs in enumerate(tqdm(dataloader)):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # perturb inputs
            augmented_inputs = perturb_inputs(inputs, tokenizer, aug_wordswap)
            # get predictions
            with torch.no_grad():
                outputs = model(**augmented_inputs)
            predictions = output_to_predictions(outputs, glue_type)
            # add evaluation sub-results
            metric3.add_batch(predictions=predictions, references=augmented_inputs["labels"])
        # aggregate evaluation results
        wordswap_scores_i = metric3.compute()
        for key, val in wordswap_scores_i.items():
            wordswap_scores[key].append(val)
      
    print('\n', task)
    print("original performance", original_scores)
    print("performance after substituting characters", charsub_scores)
    print("average performance after substituting characters", {key: np.mean(lst) for key, lst in charsub_scores.items()})
    print("performance after swapping words", wordswap_scores)
    print("average performance after swapping words", {key: np.mean(lst) for key, lst in wordswap_scores.items()})
    return original_scores, charsub_scores, wordswap_scores
    
    

def main():
    parser = argparse.ArgumentParser(description="Training/Eval for GLUE datasets")
    parser.add_argument("--task", "-t", type=str, help="GLUE task")
    parser.add_argument("--subset", default=False, type=bool, help="Use subset for training and valiation set evaluation for faster speed and debugging")
    
    args = parser.parse_args()
    task = args.task

    assert task in GLUE_CONFIGS
    
    if args.subset:
        model = DistilledModel("distilbert-base-uncased", task=task)
    else:
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
    
    if args.subset:
        train_dataset = torch.utils.data.Subset(train_dataset, range(256))
        val_dataset = torch.utils.data.Subset(val_dataset, range(32))
    print(f"Train Data Size: {len(train_dataset)}")
    print(f"Validation Data Size: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=data_collator)

    model = train(model, train_dataloader)
    evaluate(model, val_dataloader, tokenizer, task)


if __name__ == "__main__":
    main()
