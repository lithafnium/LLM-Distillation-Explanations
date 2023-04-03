import re
import os
import json

from copy import deepcopy
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, AutoModel, DataCollatorWithPadding, DistilBertForSequenceClassification
import numpy as np
from torch import nn
from datasets import load_dataset
from collections import OrderedDict, defaultdict
from transformers import AdamW, get_scheduler
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from tqdm import tqdm, trange
import evaluate as e
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac



from load_glue import *
from train_and_eval import Trainer
from build_explanation import predict_func, run_shap, run_lime
from model_robustness import tokenization, outputs_to_predictions, perturb_inputs

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def robust_eval_exp(model, dataloader, tokenizer, task, args):
    model.eval()
    
    # robustness - character level - replace characters
    print("replace characters randomly...")
    charsub_scores = defaultdict(list)
    for _ in range(args.perturb_times):
        pass
    return



