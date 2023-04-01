import shap
import torch 

from transformers import BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, AutoModel, DataCollatorWithPadding, DistilBertForSequenceClassification
from collections import OrderedDict
from bert_experiments import train, evaluate, train_and_eval_split

from torch.utils.data import (
    Dataset,
    DataLoader,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_shap(teacher, student, tokenizer, texts):
    bsize = 1

    model = teacher 
    def predict(x):
        # TODO: need to set indices based off of positive or negative results
        # print("x.toList(): ", x.tolist())
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
        logits = logits[:, 1]
        return logits

    explainer = shap.Explainer(predict, tokenizer)
    cur_start = 0
    out = []
    while cur_start < len(texts):
        output_dict = {}
        texts_ = texts[cur_start:cur_start + bsize]
        # print("sentence: ", texts_[0], "model index: ", text_to_model[texts_[0]])
        shap_values = explainer(texts_)
        model = student 
        output_dict["student_vals"] = shap_values.values

        shap_values = explainer(texts_)
        model = teacher
        output_dict["teacher_vals"] = shap_values.values
        output_dict["text"] = texts_

        out.append(output_dict)

        cur_start += bsize

    return out

def main(student_type, teacher_type="bert-base-uncased",task="sst2", teacher_path=None, student_path=None):
    task = "sst2"

    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
    if teacher_path:
        teacher = AutoModelForSequenceClassification.from_pretrained(teacher_path, num_labels=num_labels)
    else: 
        teacher = AutoModelForSequenceClassification.from_pretrained(teacher_type, num_labels=num_labels) 
    
    if student_path: 
        student = AutoModelForSequenceClassification.from_pretrained(student_path, num_labels=num_labels)
    else:
        student = AutoModelForSequenceClassification.from_pretrained(student_type, num_labels=num_labels)
    # print(teacher.base_model_prefix)

    # teacher = BertForSequenceClassification.from_pretrained(teacher_type, num_labels=num_labels)
    # student = DistilBertForSequenceClassification.from_pretrained(student_type, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(teacher_type)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    teacher.to(device)
    student.to(device)

    train_dataset, val_dataset, val_raw_dataset = train_and_eval_split(tokenizer, task)
    # train teacher
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=data_collator)

    # teacher = train(teacher, train_dataloader)
    
    # train student 
    train_dataset = train_dataset.remove_columns(["token_type_ids"])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=data_collator)

    # student = train(student, train_dataloader)
    eval_shap(teacher, student, tokenizer, val_raw_dataset["sentence"])
    # eval_shap(teacher, student, tokenizer, val_raw_dataset)

if __name__ == "__main__":
    main("distilbert-base-uncased")