# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" The distiller to distil the student.
    Adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
from collections import defaultdict
import math
import os
import time

import psutil
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups
from lm_seqs_dataset import LmSeqsDataset
from transformers import get_linear_schedule_with_warmup
from utils import logger

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import argparse
import json
import os
import pickle
import shutil
import random

import numpy as np
import torch

from distiller import Distiller
from lm_seqs_dataset import LmSeqsDataset
from transformers import (
    AutoConfig,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    AutoTokenizer,
    AutoModelForMaskedLM,
)
from utils import *
from datasets import load_dataset
import wandb
from models.modeling_distilbert import DistilBertForMaskedLM

class CausalDistiller:
    def __init__(
        self, params: dict, dataset: LmSeqsDataset, 
        token_probs: torch.tensor, student: nn.Module, teacher: nn.Module
    ):
        if params.is_wandb:
            run = wandb.init(
                project="Causal-BERT-Distillation", 
                entity="wuzhengx",
                name=params.run_name,
            )
            wandb.config.update(params)
        self.is_wandb = params.is_wandb
        
        logger.info("Initializing Distiller")
        self.params = params
        self.dump_path = params.dump_path
        self.multi_gpu = params.multi_gpu
        self.fp16 = params.fp16

        self.student = student
        self.teacher = teacher
        
        # causal neuron mappings.
        self.deserialized_variable_mappings = defaultdict(list)
        def load_variable_names(m):
            deserialized_variables = [] 
            for variable in m:
                deserialized_variables.append(deserialize_variable_name(variable))
            return deserialized_variables
        
        # node mappings
        # neuron mapping logic is taken from the paper's implementation, modified to fit our usecase
        with open(params.neuron_mapping) as json_file:
            logger.info(f"Loading neuron mapping {params.neuron_mapping}")
            neuron_mapping_json = json.load(json_file)

            for i in range(len(neuron_mapping_json["interchange_variable_mappings"])):
                names = neuron_mapping_json["interchange_variable_mappings"][i]

                t_deserialized_variables = load_variable_names(names["teacher_variable_names"])
                s_deserialized_variables = load_variable_names(names["student_variable_names"])
                self.deserialized_variable_mappings["teacher"].extend(t_deserialized_variables)
                self.deserialized_variable_mappings["student"].extend(s_deserialized_variables)        

        print(self.deserialized_variable_mappings)

        self.student_config = student.config
        self.vocab_size = student.config.vocab_size

        # overwrite slightly on this.
        if params.local_rank == -1:
            sampler = RandomSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
            
        if params.group_by_size:
            groups = create_lengths_groups(lengths=dataset.lengths, k=params.max_model_input_size)
            sampler = GroupedBatchSampler(sampler=sampler, group_ids=groups, batch_size=params.batch_size)
        else:
            sampler = BatchSampler(sampler=sampler, batch_size=params.batch_size, drop_last=False)

        # slower loader?
        # self.dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=dataset.batch_sequences)
        self.dataloader = DataLoader(
            dataset=dataset, batch_sampler=sampler, collate_fn=dataset.batch_sequences,
            # num_workers=8,
            # pin_memory=True,
        )

        self.temperature = params.temperature
        assert self.temperature > 0.0

        self.alpha_ce = params.alpha_ce
        self.alpha_mlm = params.alpha_mlm
        self.alpha_clm = params.alpha_clm
        self.alpha_mse = params.alpha_mse
        self.alpha_cos = params.alpha_cos
        self.alpha_causal_ce = params.alpha_causal_ce
        self.alpha_causal_cos = params.alpha_causal_cos

        self.mlm = params.mlm
        if self.mlm:
            logger.info("Using MLM loss for LM step.")
            self.mlm_mask_prop = params.mlm_mask_prop
            assert 0.0 <= self.mlm_mask_prop <= 1.0
            assert params.word_mask + params.word_keep + params.word_rand == 1.0
            self.pred_probs = torch.FloatTensor([params.word_mask, params.word_keep, params.word_rand])
            self.pred_probs = self.pred_probs.to(torch.device("cuda"), non_blocking=True) if params.n_gpu > 0 else self.pred_probs
            self.token_probs = token_probs.to(torch.device("cuda"), non_blocking=True) if params.n_gpu > 0 else token_probs
            if self.fp16:
                self.pred_probs = self.pred_probs.half()
                self.token_probs = self.token_probs.half()
        else:
            logger.info("Using CLM loss for LM step.")

        self.interchange_mlm = params.interchange_mlm
        self.interchange_prop = params.interchange_prop
        self.interchange_max_token = params.interchange_max_token # if -1 then we don't restrict on this.
        self.interchange_masked_token_only = params.interchange_masked_token_only
        self.interchange_consecutive_only = params.interchange_consecutive_only
        self.data_augment = params.data_augment
        
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.last_loss_ce = 0
        self.last_loss_mlm = 0
        self.last_loss_clm = 0
        if self.alpha_mse > 0.0:
            self.last_loss_mse = 0
        if self.alpha_cos > 0.0:
            self.last_loss_cos = 0

        self.last_loss_causal_ce = 0
        self.last_teacher_interchange_efficacy = 0
        self.last_student_interchange_efficacy = 0
        self.last_log = 0

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        if self.alpha_mse > 0.0:
            self.mse_loss_fct = nn.MSELoss(reduction="sum")
        if self.alpha_cos > 0.0:
            self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

        logger.info("--- Initializing model optimizer")
        assert params.gradient_accumulation_steps >= 1
        self.num_steps_epoch = len(self.dataloader)
        num_train_optimization_steps = (
            int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": params.weight_decay,
            },
            {
                "params": [
                    p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        logger.info(
            "------ Number of trainable parameters (student): %i"
            % sum([p.numel() for p in self.student.parameters() if p.requires_grad])
        )
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.student.parameters()]))
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon, betas=(0.9, 0.98)
        )

        warmup_steps = math.ceil(num_train_optimization_steps * params.warmup_prop)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
        )

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            logger.info(f"Using fp16 training: {self.params.fp16_opt_level} level")
            self.student, self.optimizer = amp.initialize(
                self.student, self.optimizer, opt_level=self.params.fp16_opt_level
            )
            self.teacher = self.teacher.half()

        if self.multi_gpu:
            if self.fp16:
                from apex.parallel import DistributedDataParallel

                logger.info("Using apex.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(self.student)
            else:
                if params.local_rank == -1:
                    logger.info("Using nn.DataParallel for the teacher model.")
                    # teacher also use multi-GPU.
                    self.teacher = torch.nn.DataParallel(self.teacher)
                    self.teacher.to(torch.device("cuda")) # no rank is needed!

                    logger.info("Using nn.DataParallel for the student model.")
                    self.student = torch.nn.DataParallel(self.student)
                    self.student.to(torch.device("cuda")) # no rank is needed!
                else:
                
                    from torch.nn.parallel import DistributedDataParallel

                    logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
                    self.student = DistributedDataParallel(
                        self.student,
                        device_ids=[params.local_rank],
                        output_device=params.local_rank,
                        find_unused_parameters=True,
                    )

        self.is_master = params.is_master
        
        
    def prepare_batch_mlm(self, batch):
        """
        Prepare the batch: from the token_ids and the lengths, compute the attention mask and the masked label for MLM.

        Input:
        ------
            batch: `Tuple`
                token_ids: `torch.tensor(bs, seq_length)` - The token ids for each of the sequence. It is padded.
                lengths: `torch.tensor(bs)` - The lengths of each of the sequences in the batch.

        Output:
        -------
            token_ids: `torch.tensor(bs, seq_length)` - The token ids after the modifications for MLM.
            attn_mask: `torch.tensor(bs, seq_length)` - The attention mask for the self-attention.
            mlm_labels: `torch.tensor(bs, seq_length)` - The masked language modeling labels. There is a -100 where there is nothing to predict.
        """
        token_ids, lengths = batch
        token_ids, lengths = self.round_batch(x=token_ids, lengths=lengths)
        assert token_ids.size(0) == lengths.size(0)

        attn_mask = torch.arange(token_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]

        bs, max_seq_len = token_ids.size()
        mlm_labels = token_ids.new(token_ids.size()).copy_(token_ids)

        x_prob = self.token_probs[token_ids.flatten()]
        n_tgt = math.ceil(self.mlm_mask_prop * lengths.sum().item())
        tgt_ids = torch.multinomial(x_prob / x_prob.sum(), n_tgt, replacement=False)
        pred_mask = torch.zeros(
            bs * max_seq_len, dtype=torch.bool, device=token_ids.device
        )  # previously `dtype=torch.uint8`, cf pytorch 1.2.0 compatibility
        pred_mask[tgt_ids] = 1
        pred_mask = pred_mask.view(bs, max_seq_len)

        pred_mask[token_ids == self.params.special_tok_ids["pad_token"]] = 0

        # mask a number of words == 0 [8] (faster with fp16)
        if self.fp16:
            n1 = pred_mask.sum().item()
            if n1 > 8:
                pred_mask = pred_mask.view(-1)
                n2 = max(n1 % 8, 8 * (n1 // 8))
                if n2 != n1:
                    pred_mask[torch.nonzero(pred_mask).view(-1)[: n1 - n2]] = 0
                pred_mask = pred_mask.view(bs, max_seq_len)
                assert pred_mask.sum().item() % 8 == 0, pred_mask.sum().item()

        _token_ids_real = token_ids[pred_mask]
        _token_ids_rand = _token_ids_real.clone().random_(self.vocab_size)
        _token_ids_mask = _token_ids_real.clone().fill_(self.params.special_tok_ids["mask_token"])
        probs = torch.multinomial(self.pred_probs, len(_token_ids_real), replacement=True).to(_token_ids_real.device, non_blocking=True)
        _token_ids = (
            _token_ids_mask * (probs == 0).long()
            + _token_ids_real * (probs == 1).long()
            + _token_ids_rand * (probs == 2).long()
        )
        token_ids = token_ids.masked_scatter(pred_mask, _token_ids)

        mlm_labels[~pred_mask] = -100  # previously `mlm_labels[1-pred_mask] = -1`, cf pytorch 1.2.0 compatibility

        # sanity checks
        assert 0 <= token_ids.min() <= token_ids.max() < self.vocab_size

        return token_ids, attn_mask, mlm_labels, pred_mask

    def prepare_batch_clm(self, batch):
        """
        Prepare the batch: from the token_ids and the lengths, compute the attention mask and the labels for CLM.

        Input:
        ------
            batch: `Tuple`
                token_ids: `torch.tensor(bs, seq_length)` - The token ids for each of the sequence. It is padded.
                lengths: `torch.tensor(bs)` - The lengths of each of the sequences in the batch.

        Output:
        -------
            token_ids: `torch.tensor(bs, seq_length)` - The token ids after the modifications for MLM.
            attn_mask: `torch.tensor(bs, seq_length)` - The attention mask for the self-attention.
            clm_labels: `torch.tensor(bs, seq_length)` - The causal language modeling labels. There is a -100 where there is nothing to predict.
        """
        token_ids, lengths = batch
        token_ids, lengths = self.round_batch(x=token_ids, lengths=lengths)
        assert token_ids.size(0) == lengths.size(0)

        attn_mask = torch.arange(token_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]
        clm_labels = token_ids.new(token_ids.size()).copy_(token_ids)
        clm_labels[~attn_mask] = -100  # previously `clm_labels[1-attn_mask] = -1`, cf pytorch 1.2.0 compatibility

        # sanity checks
        assert 0 <= token_ids.min() <= token_ids.max() < self.vocab_size

        return token_ids, attn_mask, clm_labels

    def round_batch(self, x: torch.tensor, lengths: torch.tensor):
        """
        For float16 only.
        Sub-sample sentences in a batch, and add padding, so that each dimension is a multiple of 8.

        Input:
        ------
            x: `torch.tensor(bs, seq_length)` - The token ids.
            lengths: `torch.tensor(bs, seq_length)` - The lengths of each of the sequence in the batch.

        Output:
        -------
            x:  `torch.tensor(new_bs, new_seq_length)` - The updated token ids.
            lengths: `torch.tensor(new_bs, new_seq_length)` - The updated lengths.
        """
        if not self.fp16 or len(lengths) < 8:
            return x, lengths

        # number of sentences == 0 [8]
        bs1 = len(lengths)
        bs2 = 8 * (bs1 // 8)
        assert bs2 > 0 and bs2 % 8 == 0
        if bs1 != bs2:
            idx = torch.randperm(bs1)[:bs2]
            lengths = lengths[idx]
            slen = lengths.max().item()
            x = x[idx, :slen]
        else:
            idx = None

        # sequence length == 0 [8]
        ml1 = x.size(1)
        if ml1 % 8 != 0:
            pad = 8 - (ml1 % 8)
            ml2 = ml1 + pad
            if self.mlm:
                pad_id = self.params.special_tok_ids["pad_token"]
            else:
                pad_id = self.params.special_tok_ids["unk_token"]
            padding_tensor = torch.zeros(bs2, pad, dtype=torch.long, device=x.device).fill_(pad_id)
            x = torch.cat([x, padding_tensor], 1)
            assert x.size() == (bs2, ml2)

        assert x.size(0) % 8 == 0
        assert x.size(1) % 8 == 0
        return x, lengths

    def prepare_interchange_mask(
        self,
        lengths,
        pred_mask
    ):        
        # params
        interchange_prop = self.interchange_prop
        interchange_max_token = self.interchange_max_token # if -1 then we don't restrict on this.
        
        interchange_mask = torch.zeros_like(pred_mask, dtype=torch.bool)

        batch_size, max_seq_len = pred_mask.shape[0], pred_mask.shape[1]
        for i in range(0, batch_size):
            len = lengths[i].tolist()
            if interchange_max_token != -1:
                interchange_count = min(interchange_max_token, int(len*interchange_prop))
            else:
                interchange_count = int(len*interchange_prop)
            start_index = random.randint(0, lengths[i].len()-interchange_count)
            end_index = start_index + interchange_count
            interchange_mask[i][start_index:end_index] = 1

        return interchange_mask
    
    # def prepare_interchange_position(
    #     self, 
    #     lengths, dual_lengths,
    #     pred_mask, dual_pred_mask,
    # ):
    #     interchange_prop = self.interchange_prop
    #     batch_size = lengths.shape[0]
    #     interchange_position = []
    #     for i in range(0, batch_size):
    #         min_len = min(lengths[i].tolist(), dual_lengths[i].tolist())
    #         interchange_count = int(min_len*interchange_prop)
    #         start_index = random.randint(0, lengths[i].tolist()-interchange_count)
    #         end_index = start_index + interchange_count
    #         dual_start_index = random.randint(0, dual_lengths[i].tolist()-interchange_count)
    #         dual_end_index = dual_start_index + interchange_count
    #         interchange_position += [[start_index, end_index, dual_start_index, dual_end_index]]
    #     interchange_position = torch.tensor(interchange_position, dtype=torch.long).to(lengths.device, non_blocking=True)
    #     return interchange_position

    def get_interchanged_variables_mapping(self, variable_names):
        interchanged_variables_mapping = defaultdict(list)
        for i, variable in enumerate(variable_names):
            layer_index, head_index, activation_locations = variable
            interchanged_variables_mapping[layer_index].append((i, head_index, activation_locations))

        return interchanged_variables_mapping
    
    def train(self):
        """
        The real training loop.
        """
        if self.is_master:
            logger.info("Starting training")
        self.last_log = time.time()
        self.student.train()
        self.teacher.eval()

        for _ in range(self.params.n_epoch):
            if self.is_master:
                logger.info(f"--- Starting epoch {self.epoch}/{self.params.n_epoch-1}")

            iter_bar = tqdm(self.dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for batch in iter_bar:
                token_ids, lengths = batch

                if self.params.n_gpu > 0:
                    token_ids = token_ids.to(torch.device("cuda"), non_blocking=True)
                    lengths = lengths.to(torch.device("cuda"), non_blocking=True)
                
                if self.mlm:
                    token_ids, attn_mask, lm_labels, pred_mask = self.prepare_batch_mlm(
                        batch=(token_ids, lengths)
                    )
                else:
                    token_ids, attn_mask, lm_labels = self.prepare_batch_clm(batch=(token_ids, lengths))
                    
                interchange_mask, dual_interchange_mask = self.prepare_interchange_mask(
                    lengths,
                    pred_mask
                )

                self.step(
                    input_ids=token_ids, 
                    attention_mask=attn_mask, 
                    lm_labels=lm_labels,
                    interchange_mask=interchange_mask, 
                )
                iter_bar.update()
                iter_bar.set_postfix(
                    {
                        "Last_loss": f"{self.last_loss:.2f}", 
                         "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}", 
                         "Last_cf_loss": f"{self.last_loss_causal_ce:.2f}", 
                    }
                )
            iter_bar.close()

            if self.is_master:
                logger.info(f"--- Ending epoch {self.epoch}/{self.params.n_epoch-1}")
            self.end_epoch()

        if self.is_master:
            logger.info("Save very last checkpoint as `pytorch_model.bin`.")
            self.save_checkpoint(checkpoint_name="pytorch_model.bin")
            logger.info("Training is finished")

    def step(
        self, input_ids: torch.tensor, 
        attention_mask: torch.tensor, 
        lm_labels: torch.tensor,
        interchange_mask: torch.tensor,
        skip_update_iter=False,
    ):
        """
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).

        Input:
        ------
        input_ids/dual_input_ids: `torch.tensor(bs, seq_length)` - The token ids.
        attention_mask/dual_attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        lm_labels/dual_lm_labels: `torch.tensor(bs, seq_length)` - The language modeling labels (mlm labels for MLM and clm labels for CLM).
        """
        # preparing for causal distillation.
        # we randomly select the pool of neurons to interchange.
        
        teacher_variable_names = random.choice(self.deserialized_variable_mappings["teacher"])
        student_variable_names = random.choice(self.deserialized_variable_mappings["student"])

        # get variables to interchange
        teacher_interchanged_variables_mapping = self.get_interchanged_variables_mapping(teacher_variable_names)
        student_interchanged_variables_mapping = self.get_interchanged_variables_mapping(student_variable_names)        

        
        if self.mlm:
            with torch.no_grad():
                # teacher forward pass normal.
                teacher_outputs = self.teacher(
                    input_ids=input_ids, attention_mask=attention_mask
                )  # (bs, seq_length, voc_size)
                activations_teacher = get_activations(
                    teacher_outputs,
                    teacher_variable_names
                )
            student_outputs = self.student(
                input_ids=input_ids, attention_mask=attention_mask,
            )  # (bs, seq_length, voc_size)
        else:
            assert False # we are not supporting this branch!
        
        # standard losses.
        s_logits, s_hidden_states = student_outputs["logits"], student_outputs["hidden_states"]
        t_logits, t_hidden_states = teacher_outputs["logits"], teacher_outputs["hidden_states"]
        assert s_logits.size() == t_logits.size()

        # https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
        # https://github.com/peterliht/knowledge-distillation-pytorch/issues/2
        if self.params.restrict_ce_to_mask:
            mask = (lm_labels > -1).unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)
        else:
            mask = attention_mask.unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)
        s_logits_slct = torch.masked_select(s_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()

        loss_ce = (
            self.ce_loss_fct(
                nn.functional.log_softmax(s_logits_slct / self.temperature, dim=-1),
                nn.functional.softmax(t_logits_slct / self.temperature, dim=-1),
            )
            * (self.temperature) ** 2
        )
        loss = self.alpha_ce * loss_ce

        if self.alpha_mlm > 0.0:
            loss_mlm = self.lm_loss_fct(s_logits.view(-1, s_logits.size(-1)), lm_labels.view(-1))
            loss += self.alpha_mlm * loss_mlm
        if self.alpha_clm > 0.0:
            shift_logits = s_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_clm = self.lm_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss += self.alpha_clm * loss_clm

        if self.alpha_mse > 0.0:
            loss_mse = self.mse_loss_fct(s_logits_slct, t_logits_slct) / s_logits_slct.size(
                0
            )  # Reproducing batchmean reduction
            loss += self.alpha_mse * loss_mse
        if self.alpha_cos > 0.0:
            s_hidden_states = s_hidden_states[-1]  # (bs, seq_length, dim)
            t_hidden_states = t_hidden_states[-1]  # (bs, seq_length, dim)
            mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states)  # (bs, seq_length, dim)
            assert s_hidden_states.size() == t_hidden_states.size()
            dim = s_hidden_states.size(-1)

            s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
            s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
            t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
            t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)

            target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)
            loss_cos = self.cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
            loss += self.alpha_cos * loss_cos

        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_ce = loss_ce.item()
        if self.alpha_mlm > 0.0:
            self.last_loss_mlm = loss_mlm.item()
        if self.alpha_clm > 0.0:
            self.last_loss_clm = loss_clm.item()
        if self.alpha_mse > 0.0:
            self.last_loss_mse = loss_mse.item()
        if self.alpha_cos > 0.0:
            self.last_loss_cos = loss_cos.item()

        self.optimize(loss)

        self.n_sequences_epoch += input_ids.size(0)

    def optimize(self, loss, skip_update_iter=False):
        """
        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        Also update the metrics for tensorboard.
        """
        # Check for NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        if self.multi_gpu:
            loss = loss.mean()
        if self.params.gradient_accumulation_steps > 1:
            loss = loss / self.params.gradient_accumulation_steps

        if self.fp16:
            from apex import amp

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        """
        In case where we want to do two mini-steps for dual on main interchange,
        and main on dual interchange (including normal objectives), we want to
        skip the iter update, so the gradients are accumulated within the step
        which includes gradients from two mini-steps.
        """
        self.iter(skip_update_iter=skip_update_iter)

        if self.n_iter % self.params.gradient_accumulation_steps == 0:
            if self.fp16:
                nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.params.max_grad_norm)
            else:
                nn.utils.clip_grad_norm_(self.student.parameters(), self.params.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

    def iter(self, skip_update_iter=False):
        """
        Update global counts, write to tensorboard and save checkpoint.
        """
        
        if not skip_update_iter:
            self.n_iter += 1
            self.n_total_iter += 1
            if self.n_total_iter % self.params.checkpoint_interval == 0:
                self.save_checkpoint()
        
        """
        Logging is not affected by the flag skip_update_iter.
        We want to log crossway effects, and losses should be
        in the same magnitude.
        """
        if self.n_total_iter % self.params.log_interval == 0:
            self.log_tensorboard()
            self.last_log = time.time()

    def log_tensorboard(self):
        """
        Log into tensorboard. Only by the master process.
        """
        if not self.is_master:
            return
        
        if not self.is_wandb:
            return

        wandb.log(
            {
                "train/cum_avg_loss_epoch": self.total_loss_epoch / self.n_iter, 
                "train/loss": self.last_loss, 
                "train/loss_ce": self.last_loss_ce, 
            }, 
            step=self.n_total_iter
        )
        
        if self.alpha_mlm > 0.0:
            wandb.log(
                {"train/loss_mlm": self.last_loss_mlm}, 
                step=self.n_total_iter
            )
        if self.alpha_clm > 0.0:
            wandb.log(
                {"train/loss_clm": self.last_loss_clm}, 
                step=self.n_total_iter
            )
        if self.alpha_mse > 0.0:
            wandb.log(
                {"train/loss_mse": self.last_loss_mse}, 
                step=self.n_total_iter
            )
        if self.alpha_cos > 0.0:
            wandb.log(
                {"train/loss_cos": self.last_loss_cos}, 
                step=self.n_total_iter
            )

        wandb.log(
            {
                "train/loss_causal_ce": self.last_loss_causal_ce,
                "train/loss_causal_cos": self.last_loss_causal_cos,
            }, 
            step=self.n_total_iter
        )
        
        wandb.log(
            {
                "train/learning_rate": self.scheduler.get_lr()[0],
                "train/memory_usage": psutil.virtual_memory()._asdict()["used"] / 1_000_000,
                "train/speed": time.time() - self.last_log,
            }, 
            step=self.n_total_iter
        )

    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """
        logger.info(f"{self.n_sequences_epoch} sequences have been trained during this epoch.")

        if self.is_master:
            self.save_checkpoint(checkpoint_name=f"model_epoch_{self.epoch}.pth")
            if self.is_wandb:
                wandb.log(
                    {
                        "epoch/loss": self.total_loss_epoch / self.n_iter, 
                        'epoch': self.epoch
                    }
                )

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0

    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """
        Save the current state. Only by the master process.
        """
        if not self.is_master:
            return
        mdl_to_save = self.student.module if hasattr(self.student, "module") else self.student
        mdl_to_save.config.save_pretrained(self.dump_path)
        state_dict = mdl_to_save.state_dict()
        torch.save(state_dict, os.path.join(self.dump_path, checkpoint_name))