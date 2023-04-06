import argparse
import json
import os
import re

import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import LayerIntegratedGradients
from lime.lime_text import LimeTextExplainer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from load_glue import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_func(model, tokenizer, task):
    def predict(x):
        inputs = tokenizer(
            x.tolist(),
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(device)

        if model.base_model_prefix != "bert":
            inputs.pop("token_type_ids")
        outputs = model(**inputs)
        logits = outputs["logits"]

        if task == "stsb":
            logits = logits[:, 0]
        else:
            logits = logits[:, 1]
        return logits

    return predict


def run_shap(model, tokenizer, dataset, task, args):
    # {idx: {'sentence': ..., 'tokens': ..., 'attributions': ..., 'base_value': ...}}
    shap_results = {}
    predict_ = predict_func(model, tokenizer, task)
    explainer = shap.Explainer(predict_, tokenizer)
    texts, labels = dataset["sentence"], dataset["label"]

    bsize = 1
    cur_start = 0
    pbar = tqdm(desc="shap texts", total=len(texts))
    while cur_start < len(texts):
        texts_ = texts[cur_start:cur_start + bsize]
        # print(texts_)
        labels_ = labels[cur_start:cur_start + bsize]
        shap_results_i = explainer(texts_)
        for j in range(bsize):
            shap_results[cur_start+j] = {'sentence': texts_[j],
                                         'tokens': shap_results_i.data[j].tolist(),
                                         'attributions': shap_results_i.values[j].tolist(),
                                         'label': labels_[j],
                                         'base_value': shap_results_i.base_values[j]}

        pbar.update(bsize)
        # move to the next batch of texts
        cur_start += bsize

    return shap_results


def run_lime(model, tokenizer, dataset, args):
    label_names = [0, 1]
    explainer = LimeTextExplainer(class_names=label_names)

    def predictor(texts):
        inputs = tokenizer(texts, return_tensors="pt",
                           truncation=True, padding=True).to(device)
        if not args.is_teacher:
            inputs.pop("token_type_ids")
        outputs = model(**inputs)
        predictions = F.softmax(outputs.logits).cpu().detach().numpy()
        return predictions

    lime_results = {}
    for i, t in enumerate(dataset):
        str_to_predict = t["sentence"]
        exp_ = explainer.explain_instance(
            str_to_predict, predictor, num_features=20, num_samples=500).as_list()
        lime_results[i] = {'sentence': str_to_predict,
                           'tokens': [tp[0] for tp in exp_],
                           'attributions': [tp[1] for tp in exp_],
                           'label': t["label"]}

    return lime_results


def run_int_grad(model, tokenizer, dataset, args):
    class ModelWrapper(nn.Module):
        def __init__(self, model, tokenizer):
            super(ModelWrapper, self).__init__()
            self.model = model
            self.tokenizer = tokenizer

        def forward(self, inputs):
            outputs = self.model(inputs)
            return torch.softmax(outputs.logits, dim=1)

        def compute_attributions(self, model_ig, text):
            self.model.eval()
            self.model.zero_grad()

            # A token used for generating token reference
            ref_token_id = self.tokenizer.pad_token_id
            # A token used as a separator between question and text and it is also added to the end of the text.
            sep_token_id = self.tokenizer.sep_token_id
            # A token used for prepending to the concatenated question-text word sequence
            cls_token_id = self.tokenizer.cls_token_id

            text_ids = self.tokenizer.encode(text, add_special_tokens=False)
            # construct input token ids
            input_ids = [cls_token_id] + text_ids + [sep_token_id]
            # construct reference token ids
            ref_input_ids = [cls_token_id] + [ref_token_id] * \
                len(text_ids) + [sep_token_id]

            input_ids = torch.tensor([input_ids], device=device)
            ref_input_ids = torch.tensor([ref_input_ids], device=device)

            # predict
            output = self.forward(input_ids)
            pred_score, pred_label_idx = torch.max(output, dim=1)

            # compute attributions and approximation delta using integrated gradients
            attributions = model_ig.attribute(inputs=input_ids,
                                              baselines=ref_input_ids,
                                              target=pred_label_idx)

            attributions = attributions.sum(dim=-1).squeeze(0)
            attributions = attributions / torch.norm(attributions)

            # print('attributions: ', attributions)
            # print('pred: ', pred_label_idx, '(', '%.2f' % pred_score, ')')

            return attributions, pred_label_idx

    model_to_attr = {
        "bert-base-uncased": "bert",
        "huawei-noah/TinyBERT_General_4L_312D": "bert",
        "distilbert-base-uncased": "distilbert",
        "distilbert_untrained": "distilbert",
        "google/mobilebert-uncased": "mobilebert",
    }

    model_wrapper = ModelWrapper(model, tokenizer)
    model_layer_ig = LayerIntegratedGradients(
        model_wrapper, getattr(model, model_to_attr[args.model_type]).embeddings.word_embeddings)

    attributions = {}
    for i, t in enumerate(tqdm(dataset)):
        sentence = t["sentence"]
        attribution, label = model_wrapper.compute_attributions(
            model_layer_ig, sentence)

        attributions[i] = {'sentence': sentence,
                           'attribution': attribution.tolist(),
                           'label': label.item()}

        # cos = nn.CosineSimilarity(dim=0)

        # attributions.append(cos(teacher_attribution, student_attribution).item())

    # TODO return something of use

    print(attributions)
    return attributions


def main():
    parser = argparse.ArgumentParser(
        description="Build explanation on validation set for models trained on GLUE datasets")
    parser.add_argument("--model_path", type=str,
                        help="Path to model to be evaluated")
    parser.add_argument("--is_teacher", action="store_true",
                        help="Whether the loaded model is a teacher model")
    parser.add_argument("--task", type=str, default="sst2")
    parser.add_argument("--exp_type", type=str, choices=[
                        "lime", "shap", "grad", "all"], default="shap", help="Choose one or both of LIME and SHAP to run")
    parser.add_argument("--debug", action="store_true",
                        help="Use validation subset and untrained model for debugging with faster speed")

    args = parser.parse_args()
    try:
        match = re.search(r"_([a-zA-Z0-9]+)\.pt", args.model_path)
        args.task = match.group(1) if match else args.task
    except TypeError as e:
        print(f"invalid model path: {e}")
    assert args.task in GLUE_CONFIGS

    if args.debug:
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2)
        print(f"model loaded - debug mode")
        args.model_type = "distilbert_untrained"
    else:
        try:
            match_teacher = re.search(r"teacher_(.+)_", args.model_path)
            match_student = re.search(r"student_(.+)_", args.model_path)
            if match_teacher:
                args.model_type = match_teacher.group(1)
                args.is_teacher = True
            else:
                args.model_type = match_student.group(1)
                args.is_teacher = False
            num_labels = 3 if args.task.startswith(
                "mnli") else 1 if args.task == "stsb" else 2
            if args.model_type == "huawei-noah-TinyBERT_General_4L_312D":
                args.model_type = "huawei-noah/TinyBERT_General_4L_312D"
            if args.model_type == "google-mobilebert-uncased":
                args.model_type = "google/mobilebert-uncased"
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_type, num_labels=num_labels)
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
    _, val_dataset, val_raw_dataset = train_and_eval_split(
        tokenizer, args.task)

    def concat(e, p1, p2):
        e["sentence"] = e[p1] + " " + e[p2]
        return e

    if args.task == "ax":
        val_raw_dataset = val_raw_dataset.map(
            lambda e: concat(e, "premise", "hypothesis"))
    elif args.task == "qnli":
        val_raw_dataset = val_raw_dataset.map(
            lambda e: concat(e, "question", "sentence"))
    elif args.task == "qqp":
        val_raw_dataset = val_raw_dataset.map(
            lambda e: conat(e, "question1", "question2"))
    elif args.task not in ["cola", "sst2"]:
        val_raw_dataset = val_raw_dataset.map(
            lambda e: concat(e, "sentence1", "sentence2"))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    val_dataset.set_format(type="torch", columns=[
                           "input_ids", "token_type_ids", "attention_mask", "labels"])
    val_dataset = val_dataset.remove_columns(["token_type_ids"])

    if args.debug:
        val_dataset = torch.utils.data.Subset(val_dataset, range(4))
        val_raw_dataset = val_raw_dataset[:4]
    print(f"Validation Data Size: {len(val_dataset)}")
    val_dataloader = DataLoader(
        val_dataset, batch_size=4, collate_fn=data_collator)

    args.model_type = args.model_type.replace("/", "-")

    if not os.path.exists(f"explanation_results"):
        os.makedirs(f"explanation_results")
    if not os.path.exists(f"explanation_results/{args.model_type}"):
        os.makedirs(f"explanation_results/{args.model_type}")

    if args.exp_type == "all" or args.exp_type == "shap":
        shap_results = run_shap(
            model, tokenizer, val_raw_dataset, args.task, args)
        with open(f'explanation_results/{args.model_type}/{args.model_type}_{args.task}_shap.json', 'w') as file1:
            json.dump(shap_results, file1)
    if args.exp_type == "all" or args.exp_type == "lime":
        if args.debug:
            val_raw_dataset = [{key: value[i] for key, value in val_raw_dataset.items(
            )} for i in range(len(val_raw_dataset["sentence"]))]
        lime_results = run_lime(model, tokenizer, val_raw_dataset, args)
        with open(f'explanation_results/{args.model_type}/{args.model_type}_{args.task}_lime.json', 'w') as file2:
            json.dump(lime_results, file2)
    if args.exp_type == "all" or args.exp_type == "grad":
        if args.debug:
            val_raw_dataset = [{key: value[i] for key, value in val_raw_dataset.items(
            )} for i in range(len(val_raw_dataset["sentence"]))]
        int_grad_results = run_int_grad(model, tokenizer, val_raw_dataset, args)
        with open(f'explanation_results/{args.model_type}/{args.model_type}_{args.task}_int_grad.json', 'w') as file3:
            json.dump(int_grad_results, file3)


if __name__ == "__main__":
    # command line for debugging: python build_explanation.py --debug --exp_type all
    # otherwise: python build_explanation.py --model_path <model path>
    main()
