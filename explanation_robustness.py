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
from train_and_eval import Trainer
from build_explanation import predict_func, run_shap, run_lime
from model_robustness import tokenization, outputs_to_predictions, perturb_inputs

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def exp_with_ptbn(model, dataloader, tokenizer, augmenter, args):
    '''record multiple explanation results given one-type-multiple-times perturbation'''
    res = defaultdict(defaultdict)
    bsize = 4
    # {"lime": {idx_times: (dict with perturbed texts, tokens, attributions, ground truth labels)},
    #  "shap": {idx_times: (dict with perturbed texts, tokens, attributions, ground truth labels)}},
    
    for i in range(args.perturb_times):
        texts, labels = [], []
        for n, inputs in enumerate(tqdm(dataloader)):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # perturb inputs
            augmented_inputs = perturb_inputs(inputs, tokenizer, augmenter)
            for j in range(bsize):
                sentence = tokenizer.decode(augmented_inputs["input_ids"][j], skip_special_tokens=True)
                texts.append(sentence)
                labels.append(inputs["labels"][j].item())
        perturbed_dataset = {"sentence": texts, "label": labels}
        if args.exp_type == "all" or args.exp_type == "shap":
            shap_results_i = run_shap(model, tokenizer, perturbed_dataset, args)
            res["shap"][i] = shap_results_i
        if args.exp_type == "all" or args.exp_type == "lime":
            if args.debug:
                perturbed_dataset = [{key: value[i] for key, value in perturbed_dataset.items()} for i in range(len(perturbed_dataset["sentence"]))]
            lime_results_i = run_lime(model, tokenizer, perturbed_dataset, args)
            res["lime"][i] = lime_results_i
            
    return res
    

def robust_eval_exp_raw(model, dataloader, tokenizer, args):
    model.eval()
    bsize = 4
    res = defaultdict(defaultdict)
    # {"charsub": ..., "wordswap": ..., "wordsub": ..., "wordsynn": ...}
    # each ... being
    # {"lime": {idx_times: (dict with perturbed texts, tokens, attributions, ground truth labels)},
    #  "shap": {idx_times: (dict with perturbed texts, tokens, attributions, ground truth labels)}},
    
    # robustness - character level - replace characters
    print("replace characters randomly...")
    aug_charsub = nac.RandomCharAug(action='substitute', aug_char_min=1, aug_char_max=1, aug_word_p=0.1, candidates=list('abcdefghijklmnopqrstuvwxyz'))
    res["charsub"] = exp_with_ptbn(model, dataloader, tokenizer, aug_charsub, args)
    
    # robustness - word level - swap adjacent word
    print("swap adjacent words...")
    aug_wordswap = naw.RandomWordAug(action="swap", aug_p=0.2)
    res["wordswap"] = exp_with_ptbn(model, dataloader, tokenizer, aug_wordswap, args)
    
    # robustness - word level - substitute word
    print("replace words randomly...")
    aug_wordsub = naw.SynonymAug(aug_min=1, aug_max=1)
    res["wordsub"] = exp_with_ptbn(model, dataloader, tokenizer, aug_wordsub, args)
    
    # robustness - word level - substitute word with synonym
    print("replace words with synonyms...")
    aug_wordsynn = naw.SynonymAug(aug_min=1, aug_max=1)
    res["wordsynn"] = exp_with_ptbn(model, dataloader, tokenizer, aug_wordsynn, args)
            
    return res
    
    
def main():
    parser = argparse.ArgumentParser(description="Perturb texts from validation set and build explanations upon them")
    parser.add_argument("--model_path", type=str, help="Path to model to be evaluated")
    parser.add_argument("--is_teacher", action="store_true", help="Whether the loaded model is a teacher model")
    parser.add_argument("--task", type=str, default="sst2")
    parser.add_argument("--exp_type", type=str, choices=["lime","shap","all"], default="shap", help="Choose one or both of LIME and SHAP to run")
    parser.add_argument("--perturb_times", default=10, type=int, help="Number of times to perturb an instance to check robustness")
    parser.add_argument("--debug", action="store_true", help="Use validation subset and untrained model for debugging with faster speed")
    
    args = parser.parse_args()
    if args.debug:
        args.perturb_times = 2
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
                args.is_teacher = True
            else:
                model_type = match_student.group(1)
                args.is_teacher = False
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
        val_dataset = torch.utils.data.Subset(val_dataset, range(4))
        val_raw_dataset = val_raw_dataset[:4]
    print(f"Validation Data Size: {len(val_dataset)}")
    val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=data_collator)
    
    if not os.path.exists(f"explanation_robustness_results"):
        os.makedirs(f"explanation_robustness_results")
    if not os.path.exists(f"explanation_robustness_results/{model_type}"):
        os.makedirs(f"explanation_robustness_results/{model_type}")
        
    exps_with_perturbations = robust_eval_exp_raw(model, val_dataloader, tokenizer, args)
    print(exps_with_perturbations)
    with open(f'explanation_robustness_results/{model_type}/{model_type}_{args.task}.json', 'w') as file:
            json.dump(exps_with_perturbations, file)
            
if __name__ == "__main__":
    # command line for debugging: python explanation_robustness.py --debug --exp_type all
    # otherwise: python explanation_robustness.py --model_path <model path>
    main()


