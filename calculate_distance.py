import argparse
import json 
import math
import numpy as np
from numpy import dot
from numpy.linalg import norm
from collections import defaultdict
tasks = ["wnli", "cola", "sst2", "mrpc", "rte", "qnli"]

# tasks = ["rte"]

baseline = "bert-base-uncased"
student_models = [
    "distilbert-base-uncased", 
    "google-mobilebert-uncased", 
    "huawei-noah-TinyBERT_General_4L_312D"
]

def cos_sim(a, b): 
    return dot(a, b)/(norm(a)*norm(b))

def calc_lime(task, student):
    print("TASK: ", task, "STUDENT: ", student)

    baseline_shap_values = open(f"explanation_results/bert-base-uncased/bert-base-uncased_{task}_lime.json")
    baseline_shap_values = json.load(baseline_shap_values)

    student_shap_values = open(f"explanation_results/{student}/{student}_{task}_lime.json")
    student_shap_values = json.load(student_shap_values)

    # TODO: fix token mapping, can lead to wrong results

    total_cos_sim = 0
    total_l2 = 0
    n = 0
    for k, v in baseline_shap_values.items():
        teacher_obj = baseline_shap_values[k]
        student_obj = student_shap_values[k]

        teacher_tokens = teacher_obj["tokens_attributions"]
        student_tokens = student_obj["tokens_attributions"]

        teacher_attributions = [] 
        student_attributions = [] 

        for t in teacher_tokens:
            if t not in student_tokens:
                continue
            teacher_attributions.append(teacher_tokens[t])
            student_attributions.append(student_tokens[t])

        teacher_attributions = np.array(teacher_attributions)
        student_attributions = np.array(student_attributions)
        # if task == "rte" and student == "google-mobilebert-uncased":
        #     print(teacher_attributions, student_attributions)
        cs = cos_sim(teacher_attributions, student_attributions)
        if not math.isnan(cs):
            n += 1
            total_cos_sim += cs
        total_l2 += np.linalg.norm(teacher_attributions - student_attributions)
    
    print(total_cos_sim / n, total_l2 / len(baseline_shap_values))

def calc_shap(task, student): 
    baseline_shap_values = open(f"explanation_results/bert-base-uncased/bert-base-uncased_{task}_shap.json")
    baseline_shap_values = json.load(baseline_shap_values)

    student_shap_values = open(f"explanation_results/{student}/{student}_{task}_shap.json")
    student_shap_values = json.load(student_shap_values)

    total_cos_sim = 0
    total_l2 = 0
    n = len(baseline_shap_values)
    for k, v in baseline_shap_values.items():
        teacher_obj = baseline_shap_values[k]
        student_obj = student_shap_values[k]

        teacher_attributions = np.array(teacher_obj["attributions"])
        student_attributions = np.array(student_obj["attributions"])
        
        total_cos_sim += cos_sim(teacher_attributions, student_attributions)
        total_l2 += np.linalg.norm(teacher_attributions - student_attributions)
    
    print("TASK: ", task, "STUDENT: ", student)
    print(total_cos_sim / n, total_l2 / n)
    

def calc_ig(task, student): 
    baseline_shap_values = open(f"explanation_results/bert-base-uncased/bert-base-uncased_{task}_int_grad.json")
    baseline_shap_values = json.load(baseline_shap_values)

    student_shap_values = open(f"explanation_results/{student}/{student}_{task}_int_grad.json")
    student_shap_values = json.load(student_shap_values)

    total_cos_sim = 0
    total_l2 = 0
    n = len(baseline_shap_values)
    for k, v in baseline_shap_values.items():
        teacher_obj = baseline_shap_values[k]
        student_obj = student_shap_values[k]

        teacher_attributions = np.array(teacher_obj["attribution"])
        student_attributions = np.array(student_obj["attribution"])
        total_cos_sim += cos_sim(teacher_attributions, student_attributions)
        total_l2 += np.linalg.norm(teacher_attributions - student_attributions)
    
    print("TASK: ", task, "STUDENT: ", student)
    print(total_cos_sim / n, total_l2 / n)



if __name__ == "__main__":
    out = {}
    for task in tasks: 

        for s in student_models:
            lime_diff = calc_lime(task, s)
            # shap_diff = calc_shap(task, s) 
            # ig_diff = calc_ig(task, s)
        