import os
import json 
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict

folder = "explanation_results"
baseline = "bert-base-uncased"
student_models = [
    "distilbert-base-uncased", 
    "google-mobilebert-uncased", 
    "huawei-noah-TinyBERT_General_4L_312D"
]

exp_types = ["lime", "shap", "int_grad"]
EXP_NAME_MAPPING = {"lime": "LIME", "shap": "SHAP", "int_grad": "IntGrad"}

tasks = ["wnli", "cola", "sst2", "mrpc", "rte", "qnli"]
TASK_NAME_MAPPING = {"wnli": "WNLI", 
                     "cola": "CoLA", 
                     "sst2": "SST-2", 
                     "mrpc": "MRPC", 
                     "rte": "RTE", 
                     "qnli": "QNLI"}

metrics = ["cos_sim", "l2"]
METRIC_NAME_MAPPING = {"l2": "L2 Distance", 'cos_sim': "Cosine Similarity"}


def tk_attr_map(json_dict, idx):
    return json_dict[f"{idx}"]['tokens_attributions']
  
def cos_sim(list1, list2):
    return np.dot(list1, list2) / (np.linalg.norm(list1)*np.linalg.norm(list2))


def combine_exp_res(dict_bert, dict_distilbert, dict_mobilebert, dict_tinybert, exp_type):
    N = len(dict_bert)
    res_cos_sim = OrderedDict({"distilbert": [], "mobilebert": [], "tinybert": []})
    res_l2_dist = OrderedDict({"distilbert": [], "mobilebert": [], "tinybert": []})

    for i in range(N):
        if exp_type=='lime':
            bert_map_i = tk_attr_map(dict_bert, i)
            features_i = [key for key, val in bert_map_i.items()]
            bert_attr_i = np.array([val for key, val in bert_map_i.items()])
        elif exp_type=='shap':
            bert_attr_i = np.array(dict_bert[f'{i}']['attributions'])
        elif exp_type=='int_grad':
            bert_attr_i = np.array(dict_bert[f'{i}']['attribution'])

        if exp_type=='lime':
            distilbert_map_i = tk_attr_map(dict_distilbert, i)
            distilbert_attr_i = np.array([distilbert_map_i[ft] if ft in distilbert_map_i else 0 for ft in features_i])
            mobilebert_map_i = tk_attr_map(dict_mobilebert, i)
            mobilebert_attr_i = np.array([mobilebert_map_i[ft] if ft in mobilebert_map_i else 0 for ft in features_i])
            tinybert_map_i = tk_attr_map(dict_tinybert, i)
            tinybert_attr_i = np.array([tinybert_map_i[ft] if ft in tinybert_map_i else 0 for ft in features_i])
        elif exp_type=='shap':
            distilbert_attr_i = np.array(dict_distilbert[f'{i}']['attributions'])
            mobilebert_attr_i = np.array(dict_mobilebert[f'{i}']['attributions'])
            tinybert_attr_i = np.array(dict_tinybert[f'{i}']['attributions'])
        elif exp_type=='int_grad':
            distilbert_attr_i = np.array(dict_distilbert[f'{i}']['attribution'])
            mobilebert_attr_i = np.array(dict_mobilebert[f'{i}']['attribution'])
            tinybert_attr_i = np.array(dict_tinybert[f'{i}']['attribution'])
        
        cos_sim_distilbert = cos_sim(bert_attr_i, distilbert_attr_i)
        l2_dist_disstilbert = np.linalg.norm(bert_attr_i-distilbert_attr_i)
        res_cos_sim['distilbert'].append(cos_sim_distilbert)
        res_l2_dist['distilbert'].append(l2_dist_disstilbert)
        cos_sim_mobilebert = cos_sim(bert_attr_i, mobilebert_attr_i)
        l2_dist_mobilebert = np.linalg.norm(bert_attr_i-mobilebert_attr_i)
        res_cos_sim['mobilebert'].append(cos_sim_mobilebert)   
        res_l2_dist['mobilebert'].append(l2_dist_mobilebert)    
        cos_sim_tinybert = cos_sim(bert_attr_i, tinybert_attr_i)
        l2_dist_tinybert = np.linalg.norm(bert_attr_i-tinybert_attr_i)
        res_cos_sim['tinybert'].append(cos_sim_tinybert)
        res_l2_dist['tinybert'].append(l2_dist_tinybert)
    
    return {'cos_sim': res_cos_sim, 'l2': res_l2_dist}


def grapher(attr_dict, task, exp_type, metric):
    df = pd.DataFrame(attr_dict)
    plt.violinplot([df["distilbert"][~np.isnan(df["distilbert"])], 
                    df["mobilebert"][~np.isnan(df["mobilebert"])], 
                    df["tinybert"][~np.isnan(df["tinybert"])]])
    plt.xticks([1,2,3], ["distilbert", "mobilebert", "tinybert"])
    plt.ylabel(f"{METRIC_NAME_MAPPING[metric]}")
    plt.title(f"{EXP_NAME_MAPPING[exp_type]} {METRIC_NAME_MAPPING[metric]}, student v.s. BERT: {TASK_NAME_MAPPING[task]} Task")
    #plt.show()
    if not os.path.exists(f"{folder}/comparison_plots/{task}"):
        os.makedirs(f"{folder}/comparison_plots/{task}")
    plt.savefig(f"{folder}/comparison_plots/{task}/{task}_{exp_type}_{metric}.png")
    plt.close()


def record_statistics(attr_dict, task, exp_type, metric, file):
    distilbert_avg, distilbert_median = np.nanmean(attr_dict['distilbert']), np.nanmedian(attr_dict['distilbert'])
    mobilebert_avg, mobilebert_median = np.nanmean(attr_dict['mobilebert']), np.nanmedian(attr_dict['mobilebert'])
    tinybert_avg, tinybert_median = np.nanmean(attr_dict['tinybert']), np.nanmedian(attr_dict['tinybert'])
    N = len(attr_dict['distilbert'])
    N_distilbert = N - sum(np.isnan(attr_dict['distilbert']))
    N_mobilebert = N - sum(np.isnan(attr_dict['mobilebert']))
    N_tinybert = N - sum(np.isnan(attr_dict['tinybert']))
    with open(file, 'a') as f:
        f.write(f"Task: {task}, Explanation Type: {exp_type}\n")
        f.write(f"Distilbert {METRIC_NAME_MAPPING[metric]} - AVG {distilbert_avg:.5f}, MEDIAN {distilbert_median:.5f}, N {N_distilbert}\n")
        f.write(f"Mobilebert {METRIC_NAME_MAPPING[metric]} - AVG {mobilebert_avg:.5f}, MEDIAN {mobilebert_median:.5f}, N {N_mobilebert}\n")
        f.write(f"Tinybert {METRIC_NAME_MAPPING[metric]} - AVG {tinybert_avg:.5f}, MEDIAN {tinybert_median:.5f}, N {N_tinybert}\n\n")


def main():
    for exp_type in exp_types:
        for task in tasks:
            print(f"Analyzing {exp_type}, {task} ...")
            bert_res = json.load(open(f'{folder}/{baseline}/{baseline}_{task}_{exp_type}.json'))
            distilbert_res = json.load(open(f'{folder}/{student_models[0]}/{student_models[0]}_{task}_{exp_type}.json'))
            mobilebert_res = json.load(open(f'{folder}/{student_models[1]}/{student_models[1]}_{task}_{exp_type}.json'))
            tinybert_res = json.load(open(f'{folder}/{student_models[2]}/{student_models[2]}_{task}_{exp_type}.json'))

            combined_res = combine_exp_res(bert_res, distilbert_res, mobilebert_res, tinybert_res, exp_type)
            for metric in metrics:
                record_statistics(combined_res[metric], task, exp_type, metric, "explanation_results/comparison.txt")
                grapher(combined_res[metric], task, exp_type, metric)


if __name__ == "__main__":
    main()

