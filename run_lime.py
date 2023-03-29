import lime
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer 
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, AutoModel, DataCollatorWithPadding, DistilBertForSequenceClassification

from load_glue import load_glue_dataset

teacher_type = "bert-base-uncased"
student_type = "distilbert-base-uncased"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    task = "sst2"

    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
    teacher = BertForSequenceClassification.from_pretrained(teacher_type, num_labels=num_labels)
    student = DistilBertForSequenceClassification.from_pretrained(student_type, num_labels=num_labels)

    teacher.to(device)
    student.to(device)

    label_names = [0, 1]

    explainer_teacher = LimeTextExplainer(class_names=label_names)
    explainer_student = LimeTextExplainer(class_names=label_names)

    tokenizer = AutoTokenizer.from_pretrained(teacher_type)

    train, val, test = load_glue_dataset(tokenizer, task, return_data=True)

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

    for i, t in enumerate(test):
        str_to_predict = t["sentence"]
        exp_teacher = explainer_teacher.explain_instance(str_to_predict, teacher_predictor, num_features=20, num_samples=2000)
        exp_teacher.save_to_file(f"example-{i}-teacher.html") 

        exp_student = explainer_student.explain_instance(str_to_predict, student_predictor, num_features=20, num_samples=2000)
        exp_student.save_to_file(f"example-{i}-student.html") 
        input()   

if __name__ == "__main__":
    main()
    

