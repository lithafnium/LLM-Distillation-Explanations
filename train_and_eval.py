from copy import deepcopy
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, AutoModel, DataCollatorWithPadding, DistilBertForSequenceClassification
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

from load_glue import train_and_eval_split

import argparse


NUM_EPOCHS = 3
glue_type = "cola"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(
        self,
        lr=5e-5,
        batch_size=4,
        epochs=3,
        task="sst2", 
        teacher_type="bert-base-uncased",
        student_type="distilbert-base-uncased",
        train_teacher=False, 
        train_student=False
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.task = task
        self.teacher_type= teacher_type
        self.student_type= student_type

        self.train_teacher = train_teacher 
        self.train_student = train_student


    def train(self, model, dataloader, save_path="model.pt"):
        num_training_steps = NUM_EPOCHS * len(dataloader)
        optimizer = AdamW(model.parameters(), lr=5e-5)

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        print("traiing")
        model.train()
        for step in trange(self.epochs):
            for n, inputs in enumerate(tqdm(dataloader)):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model_output_dict = model(**inputs)

                loss = model_output_dict["loss"]

                loss.backward() 
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        torch.save(model.state_dict(), save_path)
        return model

    def evaluate(self, model, dataloader, task):
        metric = e.load("glue", task)
        model.eval()
        
        for n, inputs in enumerate(tqdm(dataloader)):
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits
            if glue_type != "stsb":
                predictions = torch.argmax(logits, dim=-1)
            else:
                predictions = logits[:, 0]

            metric.add_batch(predictions=predictions, references=inputs["labels"])

        print(task, metric.compute())

    def train_and_eval(self):
        print("loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        print("loading data for task ", self.task)
        train_dataset, val_dataset, val_raw_dataset = train_and_eval_split(tokenizer, self.task)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        num_labels = 3 if self.task.startswith("mnli") else 1 if self.task=="stsb" else 2


        teacher = AutoModelForSequenceClassification.from_pretrained(self.teacher_type, num_labels=num_labels) 
        teacher.to(device)

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, collate_fn=data_collator)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, collate_fn=data_collator)

        if self.train_teacher:
            print("training teacher...")
            teacher = self.train(teacher, train_dataloader, save_path=f"teacher_{self.teacher_type}_{self.task}.pt")

        student = AutoModelForSequenceClassification.from_pretrained(self.student_type, num_labels=num_labels)
        student.to(device)

        train_dataset = train_dataset.remove_columns(["token_type_ids"])
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, collate_fn=data_collator)
        
        if self.train_student:
            print("training student...")
            student = self.train(student, train_dataloader, save_path=f"student_{self.student_type}_{self.task}.pt")

        self.evaluate(teacher, val_dataloader, self.task)
        self.evaluate(student, val_dataloader, self.task)

        return teacher, student



    


