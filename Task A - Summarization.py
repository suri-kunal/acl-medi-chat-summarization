#!/usr/bin/env python
# coding: utf-8
# %%

# %%


# Imports
from pathlib import Path

import datasets as ds
import nltk
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.optim import AdamW
from tqdm.auto import tqdm

import transformers
from filelock import FileLock
import huggingface_hub as hf_hub

from transformers import AutoConfig, \
                         AutoModelForSeq2SeqLM, \
                         AutoTokenizer, \
                         BartTokenizer, \
                         DataCollatorForSeq2Seq, \
                         SchedulerType, \
                         get_scheduler, \
                         set_seed, \
                         get_linear_schedule_with_warmup, \
                         SchedulerType                         

from bert_score import score
import evaluate
import wandb
import pandas as pd
import random

import re
import os
import config as code_config

from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
import math
import json

# %%


os.environ["WANDB_API_KEY"] = code_config.WANDB_API
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "online"
hf_hub.login(code_config.HF_API)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%


try:
    nltk.data.find("tokenizers/punkt")
except LookupError as e:
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


# %%


# set_seed(code_config.SEED)


# %%


config = AutoConfig.from_pretrained(code_config.TASKA_SUMMARY_MODEL_CHECKPOINT)
tokenizer = AutoTokenizer.from_pretrained(code_config.TASKA_SUMMARY_MODEL_CHECKPOINT, \
                                          do_lower_case=True, \
                                          force_download=True)


# %%


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# %%


def preprocess_function(examples):
    seed_everything(code_config.TASKA_SUMMARY_SEED)
    
    inputs = examples[text_column]
    targets = examples[summary_column]
    
    model_inputs = \
    tokenizer(inputs, \
              padding=code_config.TASKA_SUMMARY_PADDING, \
              truncation=True, \
              max_length=code_config.TASKA_SUMMARY_MAX_SOURCE_LENGTH)
    
    labels = tokenizer(text_target=targets, \
                        padding=code_config.TASKA_SUMMARY_PADDING, \
                        truncation=True, \
                        max_length=code_config.TASKA_SUMMARY_MAX_TARGET_LENGTH)
    
    if code_config.TASKA_SUMMARY_PADDING == "max_length" and code_config.TASKA_SUMMARY_IGNORE_PAD_TOKEN_FOR_LOSS:
        labels["input_ids"] = \
        [[l if l != tokenizer.pad_token_id else -100 for l in label] for label in labels["input_ids"]]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# %%


def postprocess_text(preds,labels):
    seed_everything(code_config.TASKA_SUMMARY_SEED)
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    
    return preds, labels


# %%


section_header_mapping = \
{"fam/sochx": ["FAMILY HISTORY","SOCIAL HISTORY"], \
"genhx": ["HISTORY of PRESENT ILLNESS"], \
"pastmedicalhx": ["PAST MEDICAL HISTORY"], \
"cc": ["CHIEF COMPLAINT"], \
"pastsurgical": ["PAST SURGICAL HISTORY"], \
"allergy": ["allergy"], \
"ros": ["REVIEW OF SYSTEMS"], \
"medications": ["medications"], \
"assessment": ["assessment"], \
"exam": ["exam"], \
"diagnosis": ["diagnosis"], \
"disposition": ["disposition"], \
"plan": ["plan"], \
"edcourse": ["EMERGENCY DEPARTMENT COURSE"], \
"immunizations": ["immunizations"], \
"imaging": ["imaging"], \
"gynhx": ["GYNECOLOGIC HISTORY"], \
"procedures": ["procedures"], \
"other_history": ["other_history"], \
"labs": ["labs"]}


# %%


train_path = Path.cwd().joinpath("mediqa-chat-data","TaskA","TaskA-TrainingSet.csv")
validation_path = Path.cwd().joinpath("mediqa-chat-data","TaskA","TaskA-ValidationSet.csv")

train_df = pd.read_csv(train_path,index_col="ID")
valid_df = pd.read_csv(validation_path,index_col="ID")
merge_df = pd.concat([train_df,valid_df],axis=0,ignore_index=True)
merge_df["dialogue_wo_whitespaces"] = merge_df["dialogue"].apply(lambda x: re.sub(r'[\r\n\s]+',' ',x))
merge_df.reset_index(inplace=True)
merge_df.rename(mapper={'index':'ID'},axis=1,inplace=True)

merge_df["section_header_desription"] = \
merge_df["section_header"].apply(lambda x: " and ".join(section_header_mapping[x.lower()]))
merge_df["section_header_desription"] = merge_df["section_header_desription"].str.lower()

summary_column = "section_text"

if (code_config.TASKA_SUMMARY_DIALOGUE_W_SECTION_CODE is True) and \
    (code_config.TASKA_SUMMARY_DIALOGUE_W_SECTION_CODE_DESC is True):
    raise Exception("SUMMARY_DIALOGUE_W_SECTION_CODE and SUMMARY_DIALOGUE_W_SECTION_CODE_DESC cannot true together")
elif code_config.TASKA_SUMMARY_DIALOGUE_W_SECTION_CODE is True:
    text_column = "dialogue_w_section_header"
    merge_df[text_column] = \
    merge_df["section_header"] + f" {str(tokenizer.sep_token)} " + merge_df["dialogue_wo_whitespaces"]    
elif code_config.TASKA_SUMMARY_DIALOGUE_W_SECTION_CODE_DESC is True:
    text_column = "dialogue_w_section_header_desc"
    merge_df[text_column] = \
    merge_df["section_header_desription"] + f" {str(tokenizer.sep_token)} " + merge_df["dialogue_wo_whitespaces"]    
else:
    text_column = "dialogue_wo_whitespaces"

if code_config.TASKA_SUMMARY_SAMPLING:
    merge_df = merge_df.sample(50)

with open("TaskA-label2idx.json","r") as f:
    label2idx = json.load(f)

merge_df["label"] = merge_df["section_header"].apply(lambda x: label2idx[x])

# %%


# ######## Load Metrics from HuggingFace ########
# print('Loading ROUGE, BERTScore, BLEURT from HuggingFace')
# scorers = {
#     'rouge': (
#         evaluate.load('rouge'),
#         {'use_aggregator': False},
#         ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
#         ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
#     ),
#     'bert_scorer': (
#         evaluate.load('bertscore'),
#         {'model_type': 'microsoft/deberta-xlarge-mnli'},
#         ['precision', 'recall', 'f1'],
#         ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
#     ),
#     'bluert': (
#         evaluate.load('bleurt', config_name='BLEURT-20'),
#         {},
#         ['scores'],
#         ['bleurt']
#     ),
# }


# %%


def data_creation(df,train_idx,valid_idx,model,tokenizer):
    seed_everything(code_config.TASKA_SUMMARY_SEED)
    
    train_df = df.iloc[train_idx,:]
    valid_df = df.iloc[valid_idx,:]
    
    train_ds = ds.Dataset.from_pandas(train_df)
    valid_ds = ds.Dataset.from_pandas(valid_df)
    
    raw_dataset = ds.DatasetDict()
    raw_dataset["train"] = train_ds
    raw_dataset["valid"] = valid_ds
    
    columns = raw_dataset["train"].column_names
    processed_datasets = \
    raw_dataset.map(function=preprocess_function, \
                    batched=True, \
                    remove_columns=columns, \
                    load_from_cache_file=False, \
                    desc="Running tokenizer on dataset")
    
    train_dataset = processed_datasets["train"]
    valid_dataset = processed_datasets["valid"]
    
    label_pad_token_id = \
    -100 if code_config.TASKA_SUMMARY_IGNORE_PAD_TOKEN_FOR_LOSS else tokenizer.pad_token_id
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, \
                                       model=model, \
                                       label_pad_token_id=label_pad_token_id,)
    
    train_dataloader = DataLoader(dataset=train_dataset, \
                                  batch_size=code_config.TASKA_SUMMARY_BATCH_SIZE, \
                                  collate_fn=data_collator, \
                                  sampler=RandomSampler(train_dataset))

    valid_dataloader = DataLoader(dataset=valid_dataset, \
                                  batch_size=code_config.TASKA_SUMMARY_BATCH_SIZE, \
                                  collate_fn=data_collator, \
                                  sampler=SequentialSampler(valid_dataset))
    
    return train_dataloader,valid_dataloader


# %%


def train_summarization(model,train_dl,optimizer,lr_scheduler,epoch):
    seed_everything(code_config.TASKA_SUMMARY_SEED)
    model = model.to(device)
    model.train()
    if model.training is False:
        raise Exception("Model is not trainable")
    model.zero_grad(set_to_none=True)
    total_train_loss = 0
    for train_step,train_batch in enumerate(train_dl):
        input_ids = train_batch["input_ids"].to(device)
        attention_mask = train_batch["attention_mask"].to(device)
        labels = train_batch["labels"].to(device)
        decoder_input_ids = train_batch["decoder_input_ids"].to(device)
        output = model(input_ids=input_ids, \
                       attention_mask=attention_mask, \
                       decoder_input_ids=decoder_input_ids, \
                       labels=labels, \
                       use_cache=False, \
                       return_dict=True)
        loss = output.loss
        loss = loss / code_config.TASKA_GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.)
        total_train_loss += loss.item()
        wandb.log({'Batch/Training Loss':loss.item(), \
                   'Batch/Training Step':train_step+epoch*len(train_dl)})

        if ((train_step + 1) % code_config.TASKA_GRADIENT_ACCUMULATION_STEPS == 0) or \
        ((train_step+1) == len(train_dl)):
            optimizer.step()
            lr_scheduler.step()
            model.zero_grad(set_to_none=True)

        avg_train_loss = total_train_loss / len(train_dl)
    return avg_train_loss,model


# %%


def validate_summarization(model,valid_dl,epoch=0,log_loss=False):
    seed_everything(code_config.TASKA_SUMMARY_SEED)
    
    if model is None:
        raise Exception("Model cannot be None")
    
    model = model.to(device)
    model.eval()
    if model.training is True:
        raise Exception("Model should not be trainable")
    
    total_loss = 0
    for valid_step, valid_batch in enumerate(valid_dl):
        input_ids = valid_batch["input_ids"].to(device)
        attention_mask = valid_batch["attention_mask"].to(device)
        labels = valid_batch["labels"].to(device)
        decoder_input_ids = valid_batch["decoder_input_ids"].to(device)
        
        with torch.no_grad():
            output = model(input_ids=input_ids, \
                           attention_mask=attention_mask, \
                           decoder_input_ids=decoder_input_ids, \
                           labels=labels, \
                           use_cache=False, \
                           return_dict=True)
            loss = output.loss
            total_loss += loss.item()
            
            if log_loss is True:
                wandb.log({'Batch/Validation Loss':loss.item(), \
                           'Batch/Validation Step':valid_step+epoch*len(valid_dl)})
        
    avg_loss = total_loss / len(valid_dl)
    
    return avg_loss, model   


# %%


def generate_summarization(model,valid_dl):
    seed_everything(code_config.TASKA_SUMMARY_SEED)
    
    generated_data_list, reference_list = [], []
    
    if model is None:
        raise Exception("Model cannot be None")
    
    model = model.to(device)
    model.eval()
    if model.training is True:
        raise Exception("Model should not be trainable")
    
    gen_kwargs = {
        "max_length": code_config.TASKA_SUMMARY_MAX_TARGET_LENGTH, \
        "min_length": code_config.TASKA_SUMMARY_MIN_TARGET_LENGTH, \
        "num_beams": code_config.TASKA_SUMMARY_NUM_BEAMS
    }
    
    for valid_step, valid_batch in enumerate(valid_dl):
        input_ids = valid_batch["input_ids"].to(device)
        attention_mask = valid_batch["attention_mask"].to(device)
        labels = valid_batch["labels"].to(device)
        decoder_input_ids = valid_batch["decoder_input_ids"].to(device)
        
        generated_tokens = \
        model.generate(inputs=input_ids, \
                       attention_mask=attention_mask, \
                       **gen_kwargs)
        
        if isinstance(generated_tokens,tuple):
            generated_tokens = generated_tokens[0]
            
        generated_tokens_decoded = tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
        labels_w_padding_tokens = \
        [[l.item() if l != -100 else tokenizer.pad_token_id for l in label]for label in labels.cpu()]
        labels_decoded = \
        tokenizer.batch_decode(labels_w_padding_tokens,skip_special_tokens=True)
        
        generated_tokens_decoded,labels_decoded = \
        postprocess_text(generated_tokens_decoded,labels_decoded)
        
        generated_data_list.extend(generated_tokens_decoded)
        reference_list.extend(labels_decoded)
        
    return generated_data_list, reference_list


# %%


def log_validation_data(generated_tokens_list,reference_list,score_dict,split):
    
    seed_everything(code_config.TASKA_SUMMARY_SEED)
    
    table_data_dict = {f"Split_{split}_ID":np.arange(len(generated_tokens_list)), \
                      "Reference Sentence":reference_list, \
                     "Generated Sentence":generated_tokens_list}
    
    
    for k, v in score_dict.items():
        table_data_dict[k] = v
        if isinstance(v,list) or isinstance(v,np.ndarray):
            wandb.config.update({f"Final {k}":np.mean(v)})
            wandb.log({f"Final Metric/{k}":np.mean(v)})
        else:
            wandb.config.update({f"Final {k}":v})
            wandb.log({f"Final Metric/{k}":v})
    
    table_data_df = pd.DataFrame.from_dict(table_data_dict)    
    valid_table = wandb.Table(data=table_data_df)
    wandb.log({"Validation Table":valid_table})
    
    for k in score_dict:        
        wandb.log({f"Final Metric/{k}-Distribution": \
                   wandb.plot.histogram(table=valid_table,value=k,title=f"Distribution-{k}")})


# %%


def get_training_artifacts(df,train_idx,valid_idx,tokenizer):
    
    seed_everything(code_config.TASKA_SUMMARY_SEED)
    
    scheduler_name = SchedulerType.LINEAR
    model = AutoModelForSeq2SeqLM.from_pretrained(code_config.TASKA_SUMMARY_MODEL_CHECKPOINT, \
                                                  config=config, \
                                                  force_download=True, \
                                                  ignore_mismatched_sizes=True)
    
    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise Exception("Ensure that config.decoder_start_token_id is set")
    
    train_dataloader, valid_dataloader = data_creation(merge_df,train_idx,valid_idx,model,tokenizer)
    
    model = model.to(device)    
    
    no_decay = ["bias","LayerNorm.weight"]
    
    optimized_group_parameters = \
    [
        {
            "params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay":code_config.TASKA_SUMMARY_WEIGHT_DECAY
        },
        {
            "params":[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay":0.
        }
    ]
    
    optimizer = AdamW(optimized_group_parameters, \
                      lr=code_config.TASKA_SUMMARY_LEARNING_RATE, 
                      weight_decay=code_config.TASKA_SUMMARY_WEIGHT_DECAY)
    
    train_steps = \
    code_config.TASKA_SUMMARY_EPOCHS * len(train_dataloader) / code_config.TASKA_GRADIENT_ACCUMULATION_STEPS
    num_warmup_steps = code_config.TASKA_SUMMARY_NUM_WARMUP_STEPS * train_steps
    
    lr_scheduler = \
    get_scheduler(name=scheduler_name, \
                  optimizer=optimizer, \
                  num_warmup_steps=num_warmup_steps, \
                  num_training_steps=train_steps)
    
    return scheduler_name, train_dataloader, valid_dataloader, model, optimizer, train_steps, lr_scheduler


# %%


def training_loop(model_name, model, \
                  train_dl, valid_dl, \
                  optimizer, lr_scheduler):
    
    seed_everything(code_config.TASKA_SUMMARY_SEED)
    
    best_loss = np.inf
    best_model = None
    best_epoch = 0
    for epoch in tqdm(range(code_config.TASKA_SUMMARY_EPOCHS)):
        avg_train_loss, model = \
        train_summarization(model,train_dl,optimizer,lr_scheduler,epoch)
        if model.training is False:
            raise Exception("Model has to be trainable")
        new_loss, model = \
        validate_summarization(model,valid_dl,epoch=epoch,log_loss=True)
        wandb.log({"Epoch/Training Loss":avg_train_loss, \
                   "Epoch/Validation Loss":new_loss, \
                   "Epoch/Epoch":epoch})
        if new_loss < best_loss:
            if model is None:
                raise Exception("Best Model cannot be none")
            best_loss = new_loss
            best_model = model
            best_epoch = epoch
    best_model.push_to_hub(model_name)
    wandb.config.update({"Best Loss":best_loss})
    wandb.config.update({"Best Epoch":best_epoch})
    wandb.config.update({"Best Loss":best_loss})
    wandb.finish()


# %%


skf = StratifiedKFold(n_splits=code_config.TASKA_SUMMARY_N_SPLITS,shuffle=True,random_state=code_config.SEED)
for split,(train_idx,valid_idx) in enumerate(skf.split(merge_df, y=merge_df["label"])):
    
    model_name = code_config.TASKA_SUMMARY_MODEL_NAME
    model_name = f"{model_name}-{split}"
    
    scheduler_name, train_dataloader, valid_dataloader, \
    model, optimizer, train_steps, lr_scheduler = \
    get_training_artifacts(merge_df,train_idx,valid_idx,tokenizer)
    
    wandb.init(project=code_config.TASKA_SUMMARY_WANDB_PROJECT, \
               name=model_name, \
               save_code=True, \
               job_type=code_config.TASKA_SUMMARY_JOB_TYPE, \
               notes=code_config.TASKA_SUMMARY_NOTES)
    
    cfg = wandb.config
    cfg.update({
        "epochs": code_config.TASKA_SUMMARY_EPOCHS, \
        "batch_size": code_config.TASKA_SUMMARY_BATCH_SIZE, \
        "training_samples": len(train_dataloader.dataset), \
        "validation_samples": len(valid_dataloader.dataset), \
        "lr": code_config.TASKA_SUMMARY_LEARNING_RATE, \
        "scheduler": scheduler_name, \
        "max_source_length": code_config.TASKA_SUMMARY_MAX_SOURCE_LENGTH, \
        "min_target_length": code_config.TASKA_SUMMARY_MIN_TARGET_LENGTH, \
        "max_target_length": code_config.TASKA_SUMMARY_MAX_TARGET_LENGTH, \
        "gradient_accumulation_steps": code_config.TASKA_GRADIENT_ACCUMULATION_STEPS
    })
    
    training_loop(model_name, model, \
                  train_dataloader, \
                  valid_dataloader, \
                  optimizer, lr_scheduler)
