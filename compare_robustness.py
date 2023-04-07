import json
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

folder = "model_robustness_results"
baseline = "bert-base-uncased"
student_models = [
    "distilbert-base-uncased",
    "google-mobilebert-uncased",
    "huawei-noah-TinyBERT_General_4L_312D"
]
models = ["BERT", "DistilBERT", "MobileBERT", "TinyBERT"]

pert_methods = ["charsub", "wordswap", "wordsynn"]
PERT_NAME_MAPPING = {"charsub": "Character Substitution",
                     "wordswap": "Adjacent Word Swapping", "wordsynn": "Synonym Replacement"}

tasks = ["wnli", "cola", "sst2", "mrpc", "rte", "qnli"]
TASK_NAME_MAPPING = {"wnli": "WNLI",
                     "cola": "CoLA",
                     "sst2": "SST-2",
                     "mrpc": "MRPC",
                     "rte": "RTE",
                     "qnli": "QNLI"}

METRIC_NAME_MAPPING = {"matthews_correlation": "Matthews Correlation Coefficient",
                       "accuracy": "Accuracy",
                       "f1": "F1 Score",
                       "pearson": "Pearson Correlation Coefficient",
                       "spearmanr": "Spearman Correlation Coefficient"}


def combine_exp_res(dict_bert, dict_distilbert, dict_mobilebert, dict_tinybert, pert_method):
    metrics = dict_bert["default_scores"].keys()

    res = {}

    for metric in metrics:
        res_diff = []
        res_std = []

        res_diff.append(
            np.mean(dict_bert[pert_method][metric]) - dict_bert['default_scores'][metric][0])
        res_diff.append(
            np.mean(dict_distilbert[pert_method][metric]) - dict_distilbert['default_scores'][metric][0])
        res_diff.append(
            np.mean(dict_mobilebert[pert_method][metric]) - dict_mobilebert['default_scores'][metric][0])
        res_diff.append(
            np.mean(dict_tinybert[pert_method][metric]) - dict_tinybert['default_scores'][metric][0])

        res_std.append(np.std(dict_bert[pert_method][metric]))
        res_std.append(np.std(dict_distilbert[pert_method][metric]))
        res_std.append(np.std(dict_mobilebert[pert_method][metric]))
        res_std.append(np.std(dict_tinybert[pert_method][metric]))

        res[metric] = {"diff": res_diff, "std": res_std}

    return res


def grapher(pert_dict, task, pert_type):
    metrics = pert_dict.keys()
    x = np.arange(len(models))
    width = 1 / (len(metrics) + 2)  # the width of the bars

    fig, ax = plt.subplots()

    for mult, metric in enumerate(metrics):
        offset = width * (mult * 2)
        rects = ax.bar(x + offset, pert_dict[metric]["diff"], width, label=metric,
                       yerr=pert_dict[metric]["std"], alpha=0.5, ecolor='black', capsize=10)
        ax.bar_label(rects, fmt="%10.4f", padding=3)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Difference')
    ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left', ncol=3)
    # ax.hlines(y=0, xmin=0, xmax=4, linewidth=1, color='black')
    ax.set_ylim(-0.6, 0.6)
    
    plt.ylabel("Difference")
    plt.title(
        f"Effect of {PERT_NAME_MAPPING[pert_type]} Perturbations on Model Metrics on Task {TASK_NAME_MAPPING[task]}")
    if not os.path.exists(f"{folder}/comparison_plots/{task}"):
        os.makedirs(f"{folder}/comparison_plots/{task}")
    plt.savefig(
        f"{folder}/comparison_plots/{task}/{task}_{pert_type}.png")
    plt.close()


def record_statistics(pert_dict, task, pert_method, file):
    with open(file, 'a') as f:
        metrics = pert_dict.keys()

        for metric in metrics:
            metric_name = METRIC_NAME_MAPPING[metric]
            diff_dict = pert_dict[metric]['diff']
            std_dict = pert_dict[metric]['std']

            f.write(f"Task: {TASK_NAME_MAPPING[task]}, Perturbation Method: {PERT_NAME_MAPPING[pert_method]}, Metric: {metric_name}\n")
            f.write(f"BERT {metric_name} - DIFF {diff_dict[0]:.5f}, STD {std_dict[0]:.5f}\n")
            f.write(f"DistilBERT {metric_name} - DIFF {diff_dict[1]:.5f}, STD {std_dict[1]:.5f}\n")
            f.write(f"MobileBERT {metric_name} - DIFF {diff_dict[2]:.5f}, STD {std_dict[2]:.5f}\n")
            f.write(f"TinyBERT {metric_name} - DIFF {diff_dict[3]:.5f}, STD {std_dict[3]:.5f}\n")


def main():
    for task in tasks:
        bert_res = json.load(
            open(f'{folder}/{baseline}/{baseline}_{task}_robustness.json'))
        distilbert_res = json.load(
            open(f'{folder}/{student_models[0]}/{student_models[0]}_{task}_robustness.json'))
        mobilebert_res = json.load(
            open(f'{folder}/{student_models[1]}/{student_models[1]}_{task}_robustness.json'))
        tinybert_res = json.load(
            open(f'{folder}/{student_models[2]}/{student_models[2]}_{task}_robustness.json'))

        for pert_method in pert_methods:
            print(f"Analyzing {pert_method}, {task} ...")

            combined_res = combine_exp_res(
                bert_res, distilbert_res, mobilebert_res, tinybert_res, pert_method)

            # record_statistics(
                # combined_res, task, pert_method, "explanation_results/comparison.txt")
            grapher(combined_res, task, pert_method)


if __name__ == "__main__":
    main()
