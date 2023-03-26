from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel, DataCollatorWithPadding
from torch import nn
from datasets import load_dataset
from collections import OrderedDict
from transformers import AdamW, get_scheduler
import torch 
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from tqdm import tqdm, trange
import evaluate as e


from shap_utils.utils import text as get_text
import shap

max_length = 128
NUM_EPOCHS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher = "bert-base-uncased"
student = "distilbert-base-uncased"

lsm = torch.nn.LogSoftmax(dim=-1)

from datasets import disable_caching
disable_caching()



class Model(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(type, num_labels=2)
    
    def forward(self, **inputs):
        x = self.model(**inputs) 
        return x

def calculate_shapley_values(models, tokenizer, texts, text_to_model):
    bsize = 1

    model = models[0]

    def predict(x):
        # TODO: need to set indices based off of positive or negative results
        # print("x.toList(): ", x.tolist())
        inputs = tokenizer(
            x.tolist(),
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(device)
        model_output_dict = model(**inputs)
        logits = lsm(model_output_dict["logits"].detach().cpu()).numpy()
        logits = logits[:, 1]
        return logits

    sorted_dict = OrderedDict()
    explainer = shap.Explainer(predict, tokenizer)
    cur_start = 0
    while cur_start < len(texts):

        texts_ = texts[cur_start: cur_start + bsize]
        model = models[text_to_model[texts_[0]]]
        # print("sentence: ", texts_[0], "model index: ", text_to_model[texts_[0]])
        shap_values = explainer(texts_)
        shap_text = get_text(shap_values)

        for t in shap_text:
            sorted_dict[t] = shap_text[t]["span"]
        cur_start += bsize

    return sorted_dict

def train(model, dataloader):
    num_training_steps = NUM_EPOCHS * len(dataloader)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    model.train()
    for step in trange(NUM_EPOCHS):
        for n, inputs in enumerate(tqdm(dataloader)):
            inputs = {k: v.to(device) for k, v in inputs.items()}

            model_output_dict = model(**inputs)

            loss = model_output_dict["loss"]

            loss.backward() 
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    return model

def evaluate(model, dataloader):
    metric = e.load("glue", "mrpc")
    model.eval()
    val_acc = 0
    
    for n, inputs in enumerate(tqdm(dataloader)):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=inputs["labels"])

        # metric.add_batch(predictions=predictions, references=batch["labels"])
        # y_pred_softmax = torch.log_softmax(model_output_dict["logits"], dim=1)
        # _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
        # correct_pred = predictions == inputs["labels"].to(device)
        # acc = correct_pred.sum() / len(correct_pred)
        # acc = torch.round(acc * 100)


        # val_acc += acc
        # # all_logits.extend(logits)
    
    # print(val_acc / len(dataloader))
    print(metric.compute())
    # all_logits = [0 if x[0] > x[1] else 1 for x in all_logits]

    
def tokenization(tokenzier, example):
    return tokenzier(example["sentence"], 
            truncation=True,
            padding=True)
def main():
    dataset = load_dataset("glue", "sst2")
    # dataset.cleanup_cache_files()
    # print(dataset)
    # for d in dataset["test"]:
    #     print(d["label"])
    teacher_model = Model(teacher)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher)
    teacher_model.to(device)
    # student_model = Model(student)
    # teacher_tokenizer = AutoTokenizer.from_pretrained(student)
    data_collator = DataCollatorWithPadding(tokenizer=teacher_tokenizer)

    # for d in dataset["validation"]:
    #     print(d["label"])
    train_dataset = dataset["train"].map(lambda e: tokenization(teacher_tokenizer, e), batched=True)
    train_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    train_dataset = train_dataset.rename_column("label", "labels")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=data_collator)

    test_dataset = dataset["test"].map(lambda e: tokenization(teacher_tokenizer, e), batched=True)
    test_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=data_collator)
    # print(dataset)
    # teacher_model = train(teacher_model, train_dataloader)
    evaluate(teacher_model, test_dataloader)

if __name__ == "__main__":
    main()
