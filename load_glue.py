from transformers import AutoTokenizer
from datasets import load_dataset

# GLUE_CONFIGS = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']

GLUE_CONFIGS = {
    "cola": {
        "col": "sentence"
    },
    "sst2": {
        "col": "sentence"
    },
    "mrpc": {
        "example1": "sentence1", 
        "example2": "sentence2"
    },
    "stsb": {
        "example1": "sentence1", 
        "example2": "sentence2"
    },
    "rte": {
        "example1": "sentence1", 
        "example2": "sentence2"
    },
    "wnli": {
        "example1": "sentence1", 
        "example2": "sentence2"
    },
    "qqp": {
        "example1": "question1", 
        "example2": "question2"
    },
    # "mnli": {
    #     "example1": "premise", 
    #     "example2": "hypothesis"
    # },
    # "mnli_mismatched": {
    #     "example1": "premise", 
    #     "example2": "hypothesis"
    # },
    # "mnli_matched": {
    #     "example1": "premise", 
    #     "example2": "hypothesis"
    # },
    "ax": {
        "example1": "premise", 
        "example2": "hypothesis"
    },
    "qnli": {
        "example1": "question", 
        "example2": "sentence"
    }
}

def tokenization(tokenizer, col, example):
    return tokenizer(example[col], 
            truncation=True,
            padding=True) 

def tokenization_pair(tokenzier, example1, example2):
    return tokenzier(example1, example2, 
            truncation=True,
            padding=True)

# mrpc, stsb, rte, wnli: "sentence1", "sentence2"
# qqp: "question1", "question2"
# mnli, mnli_mismatched, mnli_matched, ax: "premise", "hypothesis"
# qnli: "question", "sentence"
def tokenize_pair(tokenizer, dataset, example1, example2):
    train_dataset = dataset["train"].map(lambda e: tokenization_pair(tokenizer, e[example1], e[example2]), batched=True)
    val_dataset = dataset["validation"].map(lambda e: tokenization_pair(tokenizer, e[example1], e[example2]), batched=True)
    test_dataset = dataset["test"].map(lambda e: tokenization_pair(tokenizer, e[example1], e[example2]), batched=True)
    
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    return train_dataset, val_dataset, test_dataset

# cola: "sentence"
# sst2: "sentence"
def tokenize(tokenizer, dataset, col):
    train_dataset = dataset["train"].map(lambda e: tokenization(tokenizer, col, e), batched=True)
    val_dataset = dataset["validation"].map(lambda e: tokenization(tokenizer, col, e), batched=True)
    test_dataset = dataset["test"].map(lambda e: tokenization(tokenizer, col, e), batched=True)
    
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    return train_dataset, val_dataset, test_dataset

def load_glue_dataset(tokenizer, dataset_name):
    print(f"loading {dataset_name} from GLUE...")
    dataset = load_dataset("glue", dataset_name)

    # for d in dataset["test"]:
    #     print(d["label"])
    config = GLUE_CONFIGS[dataset_name]
    print(f"config: {config}")
    print(f"tokenizing...")
    if len(config) == 2:
        train, val, test = tokenize_pair(tokenizer, dataset, **config)
    else:
        train, val, test = tokenize(tokenizer, dataset, **config)
    print(f"finished tokenizing!")
    return train, val, test

def main():
    dataset = load_dataset("glue", "qqp")
    config = GLUE_CONFIGS["qqp"]

    print(dataset)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train, val, test = tokenize_pair(tokenizer, dataset, **config)
    print(train, val, test)
  

if __name__ == "__main__":
    main()
