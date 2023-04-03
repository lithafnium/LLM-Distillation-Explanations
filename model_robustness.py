import re
import os
import json

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
from train_and_eval import *

import argparse


glue_type = "cola"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
def tokenization(tokenzier, example):
    return tokenzier(example["text"], truncation=True, padding=True)
   
   
def outputs_to_predictions(outputs, glue_type):
    '''helper function that transform model outputs to predictions'''
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
    
    
def eval_results_with_augmentation(model, dataloader, tokenizer, augmenter, metric):
    '''helper function to get evaluation results with augmentation'''
    for n, inputs in enumerate(tqdm(dataloader)):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # perturb inputs
        augmented_inputs = perturb_inputs(inputs, tokenizer, augmenter)
        # get predictions
        with torch.no_grad():
            outputs = model(**augmented_inputs)
        predictions = outputs_to_predictions(outputs, glue_type)
        # add evaluation sub-results
        metric.add_batch(predictions=predictions, references=augmented_inputs["labels"])
    # aggregate evaluation results
    charsub_scores_i = metric.compute()
    
    return charsub_scores_i
        

def robust_eval_model(model, dataloader, tokenizer, task, times=10):
    '''
    evaluate model performance in terms of metrics,
    as well as robustness as model performance when input gets perturbed
    '''
    model.eval()

    # robustness - character level - replace characters
    print("replace characters randomly...")
    charsub_scores = defaultdict(list)
    for _ in range(times):
        metric1 = e.load("glue", task)
        aug_charsub = nac.RandomCharAug(action='substitute', aug_char_min=1, aug_char_max=1, aug_word_p=0.1, candidates=list('abcdefghijklmnopqrstuvwxyz'))
        charsub_scores_i = eval_results_with_augmentation(model, dataloader, tokenizer, aug_charsub, metric1)
        for key, val in charsub_scores_i.items():
            charsub_scores[key].append(val)

    # robustness - word level - swap adjacent word
    print("swap adjacent words...")
    wordswap_scores = defaultdict(list)
    for _ in range(times):
        metric2 = e.load("glue", task)
        aug_wordswap = naw.RandomWordAug(action="swap", aug_p=0.2)
        wordswap_scores_i = eval_results_with_augmentation(model, dataloader, tokenizer, aug_wordswap, metric2)
        for key, val in wordswap_scores_i.items():
            wordswap_scores[key].append(val)
            
    # robustness - word level - substitute word
    print("replace words randomly...")
    wordsub_scores = defaultdict(list)
    for _ in range(times):
        metric3 = e.load("glue", task)
        aug_wordsub = naw.SynonymAug(aug_min=1, aug_max=1)
        wordsub_scores_i = eval_results_with_augmentation(model, dataloader, tokenizer, aug_wordsub, metric3)
        for key, val in wordsub_scores_i.items():
            wordsub_scores[key].append(val)
    
    # robustness - word level - substitute word with synonym
    print("replace words with synonyms...")
    wordsynn_scores = defaultdict(list)
    for _ in range(times):
        metric4 = e.load("glue", task)
        aug_wordsynn = naw.SynonymAug(aug_min=1, aug_max=1)
        wordsynn_scores_i = eval_results_with_augmentation(model, dataloader, tokenizer, aug_wordsynn, metric4)
        for key, val in wordsynn_scores_i.items():
            wordsynn_scores[key].append(val)
    
    print("performance after substituting characters", charsub_scores)
    print("average performance after substituting characters", {key: np.mean(lst) for key, lst in charsub_scores.items()})
    print("performance after swapping words", wordswap_scores)
    print("average performance after swapping words", {key: np.mean(lst) for key, lst in wordswap_scores.items()})
    print("performance after swapping words", wordsub_scores)
    print("average performance after substituting words", {key: np.mean(lst) for key, lst in wordsub_scores.items()})
    print("performance after swapping words", wordsynn_scores)
    print("average performance after substituting words with synonyms", {key: np.mean(lst) for key, lst in wordsynn_scores.items()})
    
    return charsub_scores, wordswap_scores, wordsub_scores, wordsynn_scores
    
    
def main():
    parser = argparse.ArgumentParser(description="Robustness evaluation for GLUE datasets")
    parser.add_argument("--model_path", type=str, help="Path to model to be evaluated")
    parser.add_argument("--is_teacher", action="store_true", help="Whether the loaded model is a teacher model")
    parser.add_argument("--student", "-s", type=str, default="distilbert-base-uncased")
    parser.add_argument("--teacher", "-t", type=str, default="bert-base-uncased")
    parser.add_argument("--task", "-t", default=None, type=str, help="GLUE task")
    parser.add_argument("--debug", action="store_true", help="Use validation subset and untrained model for debugging with faster speed")
    parser.add_argument("--perturb_times", default=10, type=int, help="Number of times to perturb an instance to check robustness")
    
    args = parser.parse_args()
    
    if args.debug:
        args.task = "sst2"
    if not args.task:
        # get task name from path name
        try:
            match = re.search(r"_([a-zA-Z0-9]+)\.pt", args.model_path)
            args.task = match.group(1) if match else args.task
        except TypeError as e:
            print(f"invalid model path: {e}")
    assert args.task in GLUE_CONFIGS
    
    if args.debug:
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        print(f"model loaded - debug mode")
        model_type = "distilbert_untrained"
    else:
        try:
            match_teacher = re.search(r"teacher_(.+)_", args.model_path)
            match_student = re.search(r"student_(.+)_", args.model_path)
            if match_teacher:
                model_type = match_teacher.group(1)
            else:
                model_type = match_student.group(1)
            num_labels = 3 if args.task.startswith("mnli") else 1 if args.task=="stsb" else 2
            model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=num_labels)
            model.load_state_dict(torch.load(args.model_path))
            print(f"model loaded")
        except FileNotFoundError as e1:
            print(f"error: {e1}")
        except RuntimeError as e2:
            print(f"error: {e2}")
        except TypeError as e3:
            print(f"error: {e3}")
    
    model.to(device)
    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(f"loading validation data for task {args.task}")
    _, val_dataset, val_raw_dataset = train_and_eval_split(tokenizer, args.task)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    val_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
    val_dataset = val_dataset.remove_columns(["token_type_ids"])
    
    if args.debug:
        val_dataset = torch.utils.data.Subset(val_dataset, range(16))
        val_raw_dataset = val_raw_dataset[:16]
    print(f"Validation Data Size: {len(val_dataset)}")
    val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=data_collator)

    charsub_scores, wordswap_scores, wordsub_scores, wordsynn_scores = robust_eval_model(model, val_dataloader, tokenizer, args.task, times=args.perturb_times)
    
    if not os.path.exists(f"model_robustness_results"):
        os.makedirs(f"model_robustness_results")
    if not os.path.exists(f"model_robustness_results/{model_type}"):
        os.makedirs(f"model_robustness_results/{model_type}")
    
    with open(f'model_robustness_results/{model_type}/{model_type}_{args.task}_robustness.json', 'w') as file:
        json.dump({"charsub": charsub_scores, "wordswap": wordswap_scores, "wordsub": wordsub_scores, "wordsynn": wordsynn_scores}, file)


if __name__ == "__main__":
    # command line for debugging: python robustness.py --debug
    # otherwise: python robustness.py --model_path <model path>
    main()
