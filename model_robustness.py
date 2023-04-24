import re
import os
import json
import inspect

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
from pattern.en import conjugate, lexeme, tag
import pattern.en


from load_glue import *
# from train_and_eval import *

import argparse

# nltk.download('punkt')
# nltk.download('wordnet')   
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk   

import ssl
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('averaged_perceptron_tagger')
def tokenization(tokenizer, example):
    return tokenizer(example["text"], truncation=True, padding=True)
   
   
def outputs_to_predictions(outputs, glue_type):
    '''helper function that transform model outputs to predictions'''
    logits = outputs.logits
    if glue_type != "stsb":
        predictions = torch.argmax(logits, dim=-1)
    else:
        predictions = logits[:, 0]
        
    return predictions


# helper class for perturbations that change verb tense
class VerbTenseAug:
    def __init__(self, tokenizer):
        self.description = "change verb tense"
        self.tokenizer = tokenizer

    def augment(self, text):
        res = []
        try:
            word_tag = tag(text, tokenize=True, encoding='utf-8', tokenizer=self.tokenizer)
        except RuntimeError:
            word_tag = tag(text, tokenize=True, encoding='utf-8', tokenizer=self.tokenizer)
        for word, pos in word_tag:
            if 'VB' in pos:
                res.append(np.random.choice(lexeme(word)))    
            else:
                res.append(word)
        return ' '.join(res)


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
    
    
def eval_results_with_augmentation(model, dataloader, tokenizer, augmenter, metric, task, perturb=True):
    '''helper function to get evaluation results with augmentation'''
    for n, inputs in enumerate(tqdm(dataloader)):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # perturb inputs
        if perturb:
            inputs = perturb_inputs(inputs, tokenizer, augmenter)
        # get predictions
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = outputs_to_predictions(outputs, task)
        # add evaluation sub-results
        metric.add_batch(predictions=predictions, references=inputs["labels"])
    # aggregate evaluation results
    charsub_scores_i = metric.compute()
    
    return charsub_scores_i
        

def robust_eval_model(model, dataloader, tokenizer, task, times=10):
    '''
    evaluate model performance in terms of metrics,
    as well as robustness as model performance when input gets perturbed
    '''
    model.eval()
    def blank():
        pass
    print("before perturbations...")
    default_scores = defaultdict(list)
    metric = e.load("glue", task) 
    scores = eval_results_with_augmentation(model, dataloader, tokenizer, blank, metric, task, perturb=False) 
    for key, val in scores.items():
        default_scores[key].append(val)

    # robsutness - word level - verb tense change
    print("change verb tense...")
    tensecg_scores = defaultdict(list)
    for _ in range(times):
        metric6 = e.load("glue", task)
        aug_tensecg = VerbTenseAug(tokenizer)
        tensecg_scores_i = eval_results_with_augmentation(model, dataloader, tokenizer, aug_tensecg, metric6, task)
        for key, val in tensecg_scores_i.items():
            tensecg_scores[key].append(val)
    
    # robustness - character level - replace characters
    print("replace characters randomly...")
    charsub_scores = defaultdict(list)
    for _ in range(times):
        metric1 = e.load("glue", task)
        aug_charsub = nac.random.RandomCharAug(action='substitute', aug_char_min=1, aug_char_max=1, aug_word_min=1, aug_word_max=1, candidates=list('abcdefghijklmnopqrstuvwxyz'))
        charsub_scores_i = eval_results_with_augmentation(model, dataloader, tokenizer, aug_charsub, metric1, task)
        for key, val in charsub_scores_i.items():
            charsub_scores[key].append(val)

    # robustness - word level - swap adjacent word
    print("swap adjacent words...")
    wordswap_scores = defaultdict(list)
    for _ in range(times):
        metric2 = e.load("glue", task)
        aug_wordswap = naw.random.RandomWordAug(action="swap", aug_min=1, aug_max=1)
        wordswap_scores_i = eval_results_with_augmentation(model, dataloader, tokenizer, aug_wordswap, metric2, task)
        for key, val in wordswap_scores_i.items():
            wordswap_scores[key].append(val)
    
    # robustness - word level - substitute word with synonym
    print("replace words with synonyms...")
    wordsynn_scores = defaultdict(list)
    for _ in range(times):
        metric3 = e.load("glue", task)
        aug_wordsynn = naw.synonym.SynonymAug(aug_min=1, aug_max=1)
        wordsynn_scores_i = eval_results_with_augmentation(model, dataloader, tokenizer, aug_wordsynn, metric3, task)
        for key, val in wordsynn_scores_i.items():
            wordsynn_scores[key].append(val)

    # robustness - character level - keyboard typo
    print("replace characters with possible keyboard error")
    kberr_scores = defaultdict(list)
    for _ in range(times):
        metric4 = e.load("glue", task)
        aug_kberr = nac.keyboard.KeyboardAug(aug_word_p=1, aug_char_min=1, aug_char_max=1)
        kberr_scores_i = eval_results_with_augmentation(model, dataloader, tokenizer, aug_kberr, metric4, task)
        for key, val in kberr_scores_i.items():
            kberr_scores[key].append(val)

    # robustness - character / word level - spelling mistake
    print("replace words with their spelling-mistake versions")
    sperr_scores = defaultdict(list)
    for _ in range(times):
        metric5 = e.load("glue", task)
        aug_sperr = naw.spelling.SpellingAug(dict_path='./utils/spelling_en.txt', aug_min=1, aug_max=1)
        sperr_scores_i = eval_results_with_augmentation(model, dataloader, tokenizer, aug_sperr, metric5, task)
        for key, val in sperr_scores_i.items():
            sperr_scores[key].append(val)
    
    
    
    print("performance before any augmentation", default_scores)
    print("performance after substituting characters", charsub_scores)
    print("average performance after substituting characters", {key: np.mean(lst) for key, lst in charsub_scores.items()})
    print("performance after swapping words", wordswap_scores)
    print("average performance after swapping words", {key: np.mean(lst) for key, lst in wordswap_scores.items()})
    print("performance after substituting words with synonyms", wordsynn_scores)
    print("average performance after substituting words with synonyms", {key: np.mean(lst) for key, lst in wordsynn_scores.items()})
    
    print("performance after substituting characters with keyboard typo", kberr_scores)
    print("average performance after substituting characters with keyboard typo", {key: np.mean(lst) for key, lst in kberr_scores.items()})
    print("performance after substituting words with their spelling-mistake versions", sperr_scores)
    print("average performance after substituting words with their spelling-mistake versions", {key: np.mean(lst) for key, lst in sperr_scores.items()})
    print("performance after changing verb tenses", tensecg_scores)
    print("average performance after changing verb tenses", {key: np.mean(lst) for key, lst in tensecg_scores.items()})

    return default_scores, charsub_scores, wordswap_scores, wordsynn_scores, kberr_scores, sperr_scores, tensecg_scores
    
    
def main():
    parser = argparse.ArgumentParser(description="Robustness evaluation for GLUE datasets")
    parser.add_argument("--model_path", type=str, help="Path to model to be evaluated")
    parser.add_argument("--is_teacher", action="store_true", help="Whether the loaded model is a teacher model")
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
            if model_type == "huawei-noah-TinyBERT_General_4L_312D":
                model_type = "huawei-noah/TinyBERT_General_4L_312D"
            if model_type == "google-mobilebert-uncased":
                model_type = "google/mobilebert-uncased"
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

    def concat(e, p1, p2):
        e["sentence"] = e[p1] + " " + e[p2]
        return e

    if args.task == "ax": 
        val_raw_dataset = val_raw_dataset.map(lambda e: concat(e, "premise", "hypothesis"))
    elif args.task == "qnli":
        val_raw_dataset = val_raw_dataset.map(lambda e: concat(e, "question", "sentence"))
    elif args.task == "qqp":
        val_raw_dataset = val_raw_dataset.map(lambda e: concat(e, "question1", "question2"))
    elif args.task not in ["cola", "sst2"]:
        val_raw_dataset = val_raw_dataset.map(lambda e: concat(e, "sentence1", "sentence2"))


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    val_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
    val_dataset = val_dataset.remove_columns(["token_type_ids"])
    
    if args.debug:
        val_dataset = torch.utils.data.Subset(val_dataset, range(4))
        val_raw_dataset = val_raw_dataset[:4]
        print(val_raw_dataset)
    print(f"Validation Data Size: {len(val_dataset)}")
    val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=data_collator)

    default_scores, charsub_scores, wordswap_scores, wordsynn_scores, kberr_scores, sperr_scores, tensecg_scores  = robust_eval_model(model, val_dataloader, tokenizer, args.task, times=args.perturb_times)
    
    model_type = model_type.replace("/", "-")

    if not os.path.exists(f"model_robustness_results_final"):
        os.makedirs(f"model_robustness_results_final")
    if not os.path.exists(f"model_robustness_results_final/{model_type}"):
        os.makedirs(f"model_robustness_results_final/{model_type}")
    
    with open(f'model_robustness_results_final/{model_type}/{model_type}_{args.task}_robustness.json', 'w') as file:
        json.dump({"default": default_scores, "charsub": charsub_scores, "wordswap": wordswap_scores, "wordsynn": wordsynn_scores, "kberr": kberr_scores, "sperr": sperr_scores, "tensecg": tensecg_scores}, file)


if __name__ == "__main__":
    # command line for debugging: python model_robustness.py --debug
    # otherwise: python model_robustness.py --model_path <model path>
    main()
