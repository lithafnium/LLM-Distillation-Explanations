import argparse

import torch
import torch.nn as nn
from captum.attr import LayerIntegratedGradients
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding

from load_glue import GLUE_CONFIGS
from train_and_eval import Trainer, train_and_eval_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STUDENT_MODEL_TO_LAYERS = {
    "huawei-noah/TinyBERT_General_4L_312D": "bert",
    "distilbert-base-uncased": "distilbert",
    "google/mobilebert-uncased": "mobilebert",
}


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, inputs):
        outputs = self.model(inputs)
        return torch.softmax(outputs.logits, dim=1)


def construct_input_ref_pair(text, tokenizer):
    # A token used for generating token reference
    ref_token_id = tokenizer.pad_token_id
    # A token used as a separator between question and text and it is also added to the end of the text.
    sep_token_id = tokenizer.sep_token_id
    # A token used for prepending to the concatenated question-text word sequence
    cls_token_id = tokenizer.cls_token_id

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    # construct input token ids
    input_ids = [cls_token_id] + text_ids + [sep_token_id]
    # construct reference token ids
    ref_input_ids = [cls_token_id] + [ref_token_id] * \
        len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device)


def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)


def compute_attributions(model, model_ig, tokenizer, text):
    model.eval()
    model.zero_grad()

    input_ids, ref_input_ids = construct_input_ref_pair(text, tokenizer)

    # predict
    output = model(input_ids)
    pred_score, pred_label_idx = torch.max(output, dim=1)

    # compute attributions and approximation delta using integrated gradients
    attributions = model_ig.attribute(inputs=input_ids,
                                      baselines=ref_input_ids,
                                      target=pred_label_idx)

    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)

    # print('attributions: ', attributions)
    # print('pred: ', pred_label_idx, '(', '%.2f' % pred_score, ')')

    return attributions


def eval_attributions(teacher, student, tokenizer, dataset, student_name):
    teacher_wrapper = ModelWrapper(teacher)
    teacher_ig = LayerIntegratedGradients(
        teacher_wrapper, teacher.bert.embeddings.word_embeddings)
    student_wrapper = ModelWrapper(student)
    student_ig = LayerIntegratedGradients(
        student_wrapper, getattr(student, STUDENT_MODEL_TO_LAYERS[student_name]).embeddings.word_embeddings)
    
    attributions = []

    for _, sentence in enumerate(tqdm(dataset["sentence"])):
        teacher_attribution = compute_attributions(
            teacher_wrapper, teacher_ig, tokenizer, sentence)
        student_attribution = compute_attributions(
            student_wrapper, student_ig, tokenizer, sentence)
        
        cos = nn.CosineSimilarity(dim=0)

        attributions.append(cos(teacher_attribution, student_attribution).item())
        
    # TODO return something of use
    return sum(attributions) / len(attributions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated Gradients")

    parser.add_argument("--student", "-s", type=str,
                        default="distilbert-base-uncased")
    parser.add_argument("--teacher", "-t", type=str,
                        default="bert-base-uncased")
    parser.add_argument("--train_teacher", "-tt", action="store_true")
    parser.add_argument("--train_student", "-ts", action="store_true")
    parser.add_argument("--task", type=str, default="sst2")

    args = parser.parse_args()
    task = args.task

    assert task in GLUE_CONFIGS

    t = Trainer(
        lr=5e-5, 
        batch_size=4,
        epochs=3,
        teacher_type=args.teacher,
        student_type=args.student,
        train_teacher=args.train_teacher,
        train_student=args.train_student,
        task=task
    )

    teacher, student = t.train_and_eval()

    tokenizer = AutoTokenizer.from_pretrained(args.teacher)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset, val_dataset, val_raw_dataset = train_and_eval_split(
        tokenizer, task)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=4, collate_fn=data_collator)
    val_dataloader = DataLoader(
        val_dataset, batch_size=4, collate_fn=data_collator)

    eval_attributions(teacher, student, tokenizer, val_raw_dataset, args.student)
