import json
import numpy as np


folder = "model_robustness_results"
baseline = "bert-base-uncased"
student_models = [
    "distilbert-base-uncased", 
    "google-mobilebert-uncased", 
    "huawei-noah-TinyBERT_General_4L_312D"
]

tasks = ["wnli", "cola", "sst2", "mrpc", "rte", "qnli"]

perturb_types = ["charsub", "wordswap", "wordsynn"]
PERTURB_NAME_MAPPING = {"charsub": "replace characters randomly", 
                        "wordswap": "swap words randomly", 
                        "wordsynn": "replace words with synonyms"}

def main():
    for task in tasks:
        print("===================================")
        for model in [baseline]+student_models:
            print("--------------------------------")
            print(f"{task}, {model}")
            result = json.load(open(f'{folder}/{model}/{model}_{task}_robustness.json'))
            for perturb_type in perturb_types:
                for metric, scores in result[perturb_type].items():
                    print(f"average {metric} after {PERTURB_NAME_MAPPING[perturb_type]}: {np.mean(scores):.4f}")


if __name__ == "__main__":
    main()