import lime
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer 
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, AutoModel, DataCollatorWithPadding, DistilBertForSequenceClassification

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from load_glue import load_glue_dataset
from bert_experiments import train, evaluate, train_and_eval_split


teacher_type = "bert-base-uncased"
student_type = "distilbert-base-uncased"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_lime(teacher, student, tokenizer, dataset):
    label_names = [0, 1]

    explainer_teacher = LimeTextExplainer(class_names=label_names)
    explainer_student = LimeTextExplainer(class_names=label_names)

    def teacher_predictor(texts):
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = teacher(**inputs)
        predictions = F.softmax(outputs.logits).cpu().detach().numpy()
        return predictions

    def student_predictor(texts):
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(device)
        inputs.pop("token_type_ids")
        outputs = student(**inputs)
        predictions = F.softmax(outputs.logits).cpu().detach().numpy()
        return predictions        

    for i, t in enumerate(dataset):
        str_to_predict = t["sentence"]
        exp_teacher = explainer_teacher.explain_instance(str_to_predict, teacher_predictor, num_features=20, num_samples=500)
        # exp_teacher.save_to_file(f"example-{i}-teacher.html") 
        teacher_list = exp_teacher.as_list()
        exp_student = explainer_student.explain_instance(str_to_predict, student_predictor, num_features=20, num_samples=500)
        # exp_student.save_to_file(f"example-{i}-student.html") 
        student_list = exp_student.as_list()

        print(teacher_list, student_list)
        input()   

def main(student_type, teacher_type="bert-base-uncased",task="sst2", teacher_path=None, student_path=None):
    task = "sst2"
    
    train_dataset, val_dataset, val_raw_dataset = train_and_eval_split(tokenizer, task)
    tokenizer = AutoTokenizer.from_pretrained(teacher_type)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
    if teacher_path:
        teacher = AutoModelForSequenceClassification.from_pretrained(teacher_path, num_labels=num_labels)
        teacher.to(device)
    else: 
        teacher = AutoModelForSequenceClassification.from_pretrained(teacher_type, num_labels=num_labels) 
        teacher.to(device)

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=data_collator)
        val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=data_collator)
        teacher = train(teacher, train_dataloader)
    if student_path: 
        student = AutoModelForSequenceClassification.from_pretrained(student_path, num_labels=num_labels)
        student.to(device)
    else:
        student = AutoModelForSequenceClassification.from_pretrained(student_type, num_labels=num_labels)
        student.to(device)

        train_dataset = train_dataset.remove_columns(["token_type_ids"])
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=data_collator)
        student = train(student, train_dataloader)

    eval_lime(teacher, student, tokenizer, val_raw_dataset["sentence"])
    # eval_shap(teacher, student, tokenizer, val_raw_dataset)

    

if __name__ == "__main__":
    main(student_type="distilbert-base-uncased")
    

