import lime
import torch
import argparse

import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer 
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, AutoModel, DataCollatorWithPadding, DistilBertForSequenceClassification

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from load_glue import load_glue_dataset

from train_and_eval import Trainer



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

    # eval_shap(teacher, student, tokenizer, val_raw_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lime Evaluation")

    parser.add_argument("--student", "-s", type=str, default="distilbert-base-uncased")
    parser.add_argument("--teacher", "-t", type=str, default="bert-base-uncased")
    parser.add_argument("--train_teacher", "-tt", action="store_true")
    parser.add_argument("--train_student", "-ts", action="store_true")
    parser.add_argument("--task", type=str, default="sst2")

    args = parser.parse_args() 

    t = Trainer(
        lr=5e-5, 
        batch_size=4,
        epochs=3,
        teacher_type=args.teacher,
        student_type=args.student,
        train_teacher=args.train_teacher,
        train_student=args.train_student,
        task=args.task
    )

    teacher, student = t.train_and_eval()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_dataset, val_dataset, val_raw_dataset = train_and_eval_split(tokenizer, self.task)



    

