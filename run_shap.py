import shap
import torch 

from transformers import BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, AutoModel, DataCollatorWithPadding, DistilBertForSequenceClassification
from collections import OrderedDict

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from train_and_eval import Trainer

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