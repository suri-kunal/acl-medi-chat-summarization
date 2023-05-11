#!/usr/bin/env python
# coding: utf-8
# %%
# Imports
import pandas as pd
import plotly
import plotly.express as px
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from pathlib import Path
import datasets as ds
import evaluate
import re
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import AutoConfig, \
                         AutoModelForSeq2SeqLM, \
                         AutoTokenizer, \
                         BartTokenizer, \
                         DataCollatorForSeq2Seq, \
                         get_scheduler, \
                         set_seed, \
                         get_linear_schedule_with_warmup, \
                         SchedulerType
import huggingface_hub as hf_hub
import numpy as np
import random
import time
import GPUtil
import wandb
import os
from tqdm import tqdm
import config as code_config
import captum
from captum.attr import LayerIntegratedGradients
import json
from sentence_transformers import SentenceTransformer, util
import nltk
from filelock import FileLock
from itertools import chain


# %%
os.environ["WANDB_API_KEY"] = code_config.WANDB_API
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "online"
hf_hub.login(code_config.HF_API)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
config = AutoConfig.from_pretrained(code_config.TASKB_SUMMARY_MODEL_CHECKPOINT)
tokenizer = AutoTokenizer.from_pretrained(code_config.TASKB_SUMMARY_MODEL_CHECKPOINT, \
                                          do_lower_case=True, \
                                          force_download=True)

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
def concatenating_section_description(section_list_of_list):
    section_list = [l for l in chain(*section_list_of_list)]
    return f" {str(tokenizer.sep_token)} ".join(section_list)


# %%
# Merging dialogue in relevant sections together
first_level_dict = dict()
first_level_description_dict = dict()

subjective = ["cc", "fam/sochx", "genhx", \
              "pastmedicalhx", "pastsurgical", \
              "gynhx", "other_history", "allergy", \
              "ros","medications","immunizations"]
subjective = [hs.upper() for hs in subjective]
subjective = list(set(subjective))
subject_list_of_list = [section_header_mapping[section.lower()] for section in subjective]
subject_description = concatenating_section_description(subject_list_of_list)

objective_exam = ["exam","imaging","labs","procedures"]
objective_exam = [oe.upper() for oe in objective_exam]
objective_exam = list(set(objective_exam))
oe_list_of_list = [section_header_mapping[section.lower()] for section in objective_exam]
objective_exam_description = concatenating_section_description(oe_list_of_list)

objective_results = ["imaging","labs", "diagnosis"]
objective_results = [obj_re.upper() for obj_re in objective_results]
objective_results = list(set(objective_results))
or_list_of_list = [section_header_mapping[section.lower()] for section in objective_results]
objective_results_description = concatenating_section_description(or_list_of_list)

assessment_and_plan = ["assessment", "plan", "disposition", \
                       "procedures", "labs", "medications", \
                       "edcourse"]
assessment_and_plan = [ap.upper() for ap in assessment_and_plan]
assessment_and_plan = list(set(assessment_and_plan))
ap_list_of_list = [section_header_mapping[section.lower()] for section in assessment_and_plan]
assessment_and_plan_description = concatenating_section_description(ap_list_of_list)

first_level_dict["subjective"] = subjective
first_level_dict["objective_exam"] = objective_exam
first_level_dict["objective_results"] = objective_results
first_level_dict["assessment_and_plan"] = assessment_and_plan

first_level_description_dict["subjective"] = subject_description
first_level_description_dict["objective_exam"] = objective_exam_description
first_level_description_dict["objective_results"] = objective_results_description
first_level_description_dict["assessment_and_plan"] = assessment_and_plan_description

# %%
merge_file = Path.cwd().joinpath("mediqa-chat-data","TaskB","TaskB-CombinedTextWithSectionDialog.csv")

merge_df = pd.read_csv(merge_file)
first_level_columns = ["subjective","objective_exam","objective_results","assessment_and_plan"]

if (code_config.TASKB_SUMMARY_DIALOGUE_W_SECTION_CODE is True) and \
    (code_config.TASKB_SUMMARY_DIALOGUE_W_SECTION_CODE_DESC is True):
    raise Exception("SUMMARY_DIALOGUE_W_SECTION_CODE and SUMMARY_DIALOGUE_W_SECTION_CODE_DESC cannot true together")
elif code_config.TASKB_SUMMARY_DIALOGUE_W_SECTION_CODE is True:
    for first_level_section in first_level_columns:
        section_str = f" {str(tokenizer.sep_token)} ".join(first_level_dict[first_level_section])
        merge_df[f"dialogue_{first_level_section}"] = \
        section_str.lower() + f" {str(tokenizer.sep_token)} " + merge_df[f"dialogue_{first_level_section}"]
elif code_config.TASKB_SUMMARY_DIALOGUE_W_SECTION_CODE_DESC is True:
    for first_level_section in first_level_columns:
        merge_df[f"dialogue_{first_level_section}"] = \
        first_level_section.lower() + \
        f" {str(tokenizer.sep_token)} " + \
        merge_df[f"dialogue_{first_level_section}"]
else:
    pass

if code_config.TASKB_SUMMARY_SAMPLING:
    merge_df = merge_df.sample(10)


# %%
try:
    nltk.data.find("tokenizers/punkt")
except LookupError as e:
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


# %%
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# %%
def preprocess_function(examples,text_column,summary_column,max_target_length):
    seed_everything(code_config.TASKB_SUMMARY_SEED)
    
    inputs = examples[text_column]
    targets = examples[summary_column]
    
    model_inputs = \
    tokenizer(inputs, \
              padding=code_config.TASKB_SUMMARY_PADDING, \
              truncation=True, \
              max_length=code_config.TASKB_SUMMARY_MAX_SOURCE_LENGTH)
    
    model_inputs["global_attention_mask"] = \
    len(model_inputs["input_ids"]) * [
        [0 for _ in range(len(model_inputs["input_ids"][0]))]
    ]
    
    for idx,mask in enumerate(model_inputs["global_attention_mask"]):
        model_inputs["global_attention_mask"][idx][0] = 1
    
    labels = tokenizer(text_target=targets, \
                        padding=code_config.TASKB_SUMMARY_PADDING, \
                        truncation=True, \
                        max_length=max_target_length)
    
    if code_config.TASKB_SUMMARY_PADDING == "max_length" and code_config.TASKB_SUMMARY_IGNORE_PAD_TOKEN_FOR_LOSS:
        labels["input_ids"] = \
        [[l if l != tokenizer.pad_token_id else -100 for l in label] for label in labels["input_ids"]]
    
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs


# %%
def postprocess_text(preds,labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    
    return preds, labels


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
def data_creation(df,train_idx,valid_idx,model,tokenizer,text_column,summary_column,max_target_length):
    train_df = df.iloc[train_idx,:]
    valid_df = df.iloc[valid_idx,:]
    
    train_ds = ds.Dataset.from_pandas(train_df)
    valid_ds = ds.Dataset.from_pandas(valid_df)
    
    raw_dataset = ds.DatasetDict()
    raw_dataset["train"] = train_ds
    raw_dataset["valid"] = valid_ds
    
    preprocess_args = {"max_target_length":max_target_length, \
                       "text_column":text_column, \
                       "summary_column":summary_column}
    
    columns = raw_dataset["train"].column_names
    processed_datasets = \
    raw_dataset.map(function=preprocess_function, \
                    batched=True, \
                    remove_columns=columns, \
                    load_from_cache_file=False, \
                    desc="Running tokenizer on dataset", \
                    fn_kwargs=preprocess_args)
    
    train_dataset = processed_datasets["train"]
    valid_dataset = processed_datasets["valid"]
    
    label_pad_token_id = -100 if code_config.TASKB_SUMMARY_IGNORE_PAD_TOKEN_FOR_LOSS else tokenizer.pad_token_id
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, \
                                       model=model, \
                                       label_pad_token_id=label_pad_token_id)
    
    train_dataloader = DataLoader(dataset=train_dataset, \
                                  batch_size=code_config.TASKB_SUMMARY_BATCH_SIZE, \
                                  collate_fn=data_collator, \
                                  sampler=RandomSampler(train_dataset))

    valid_dataloader = DataLoader(dataset=valid_dataset, \
                                  batch_size=code_config.TASKB_SUMMARY_BATCH_SIZE, \
                                  collate_fn=data_collator, \
                                  sampler=SequentialSampler(valid_dataset))
    
    return train_dataloader,valid_dataloader


# %%
def train_summarization(model,train_dl,optimizer,lr_scheduler,epoch):

    seed_everything(code_config.TASKB_SUMMARY_SEED)
    
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
        global_attention_mask = train_batch["global_attention_mask"].to(device)
        output = model(input_ids=input_ids, \
                       attention_mask=attention_mask, \
                       global_attention_mask=global_attention_mask, \
                       labels=labels, \
                       use_cache=False, \
                       return_dict=True)
        step_loss = output.loss
        total_train_loss += step_loss.item()
        wandb.log({'Batch/Training Loss':step_loss.item(), \
                   'Batch/Training Step':train_step+epoch*len(train_dl)})
        step_loss = step_loss / code_config.TASKB_GRADIENT_ACCUMULATION_STEPS
        step_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.)
        if ((train_step + 1) % code_config.TASKB_GRADIENT_ACCUMULATION_STEPS == 0) or \
            (train_step + 1 == len(train_dl)):
            optimizer.step()
            lr_scheduler.step()
            model.zero_grad(set_to_none=True)
    avg_train_loss = total_train_loss / len(train_dl)
    return avg_train_loss,model


# %%
def validate_summarization(model,valid_dl,epoch=0,log_loss=False):
    seed_everything(code_config.TASKB_SUMMARY_SEED)
    
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
        global_attention_mask = valid_batch["global_attention_mask"].to(device)
        
        with torch.no_grad():
            output = model(input_ids=input_ids, \
                           attention_mask=attention_mask, \
                           global_attention_mask=global_attention_mask, \
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
def generate_summarization(model,valid_dl,min_target_length,max_target_length):
    seed_everything(code_config.TASKB_SUMMARY_SEED)
    
    generated_data_list, reference_list = [], []
    
    if model is None:
        raise Exception("Model cannot be None")
    
    model = model.to(device)
    model.eval()
    if model.training is True:
        raise Exception("Model should not be trainable")
    
    gen_kwargs = {
        "max_length": min_target_length, \
        "min_length": max_target_length, \
        "num_beams": code_config.TASKB_SUMMARY_NUM_BEAMS
    }
    
    for valid_step, valid_batch in enumerate(valid_dl):
        input_ids = valid_batch["input_ids"].to(device)
        attention_mask = valid_batch["attention_mask"].to(device)
        labels = valid_batch["labels"].to(device)
        global_attention_mask = valid_batch["global_attention_mask"].to(device)
        
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
    
    seed_everything(code_config.TASKB_SUMMARY_SEED)
    
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
def get_training_artifacts(df,train_idx,valid_idx,tokenizer,text_column,summary_column,max_target_length):
    
    seed_everything(code_config.TASKB_SUMMARY_SEED)
    
    scheduler_name = SchedulerType.LINEAR
    model = AutoModelForSeq2SeqLM.from_pretrained(code_config.TASKB_SUMMARY_MODEL_CHECKPOINT, \
                                                  config=config, \
                                                  force_download=True, \
                                                  ignore_mismatched_sizes=True)
    
    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise Exception("Ensure that config.decoder_start_token_id is set")
    
    train_dataloader, valid_dataloader = \
    data_creation(df,train_idx,valid_idx,model,tokenizer,text_column,summary_column,max_target_length)
    
    model = model.to(device)    
    
    no_decay = ["bias","LayerNorm.weight"]
    
    optimized_group_parameters = \
    [
        {
            "params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay":code_config.TASKB_SUMMARY_WEIGHT_DECAY
        },
        {
            "params":[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay":0.
        }
    ]
    
    optimizer = AdamW(optimized_group_parameters, \
                      lr=code_config.TASKB_SUMMARY_LEARNING_RATE, 
                      weight_decay=code_config.TASKB_SUMMARY_WEIGHT_DECAY)
    
    train_steps = \
    code_config.TASKB_SUMMARY_EPOCHS * len(train_dataloader) / code_config.TASKB_GRADIENT_ACCUMULATION_STEPS
    num_warmup_steps = \
    code_config.TASKB_SUMMARY_NUM_WARMUP_STEPS * train_steps
    
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
    
    seed_everything(code_config.TASKB_SUMMARY_SEED)
    
    best_loss = np.inf
    best_model = None
    best_epoch = 0
    for epoch in tqdm(range(code_config.TASKB_SUMMARY_EPOCHS)):
        avg_train_loss,model = train_summarization(model,train_dl,optimizer,lr_scheduler,epoch)
        if model.training is False:
            raise Exception("Model has to be trainable")
        new_loss, model = \
        validate_summarization(model,valid_dl,epoch=epoch,log_loss=True)
        if new_loss < best_loss:
            if model is None:
                raise Exception("Best Model cannot be none")
            best_loss = new_loss
            best_model = model
            best_epoch = epoch
        wandb.log({"Epoch/Train Loss":avg_train_loss, \
                   "Epoch/Valid Loss":new_loss, \
                   "Epoch/Epoch":epoch})
    best_model.push_to_hub(model_name)
    wandb.config.update({"Best Loss":best_loss})
    wandb.config.update({"Best Epoch":best_epoch})
    wandb.config.update({"Best Loss":best_loss})
    wandb.finish()


# %%
max_length_by_section = dict()
min_length_by_section = dict()
df_by_section = dict()
df_by_section["subjective"] = \
merge_df[["dialogue_subjective","reference_subjective"]].dropna()
df_by_section["assessment_and_plan"] = \
merge_df[["dialogue_assessment_and_plan","reference_assessment_and_plan"]].dropna()
df_by_section["objective_results"] = \
merge_df[["dialogue_objective_results","reference_objective_results"]].dropna()
df_by_section["objective_exam"] = \
merge_df[["dialogue_objective_exam","reference_objective_exam"]].dropna()
min_length_by_section["subjective"] = code_config.TASKB_SUBJECTIVE_MIN_TARGET_LENGTH
max_length_by_section["subjective"] = code_config.TASKB_SUBJECTIVE_MAX_TARGET_LENGTH
min_length_by_section["assessment_and_plan"] = code_config.TASKB_ASSESSMENT_AND_PLAN_MIN_TARGET_LENGTH
max_length_by_section["assessment_and_plan"] = code_config.TASKB_ASSESSMENT_AND_PLAN_MAX_TARGET_LENGTH
min_length_by_section["objective_results"] = code_config.TASKB_OBJECTIVE_RESULT_MIN_TARGET_LENGTH
max_length_by_section["objective_results"] = code_config.TASKB_OBJECTIVE_RESULT_MAX_TARGET_LENGTH
min_length_by_section["objective_exam"] = code_config.TASKB_OBJECTIVE_EXAM_MIN_TARGET_LENGTH
max_length_by_section["objective_exam"] = code_config.TASKB_OBJECTIVE_EXAM_MAX_TARGET_LENGTH


# %%
def final_train_loop(data_df, train_idx, valid_idx, \
                         section, split, \
                         tokenizer, text_column, summary_column, \
                         summary_min_target_length, \
                         summary_max_target_length):
    
    model_name = code_config.TASKB_SUMMARY_MODEL_NAME
    model_name = f"{model_name}-{section}-{split}"

    scheduler_name, train_dataloader, valid_dataloader, \
    model, optimizer, train_steps, lr_scheduler = \
    get_training_artifacts(data_df, \
                           train_idx, valid_idx, \
                           tokenizer, \
                           text_column, summary_column, \
                           summary_max_target_length)

    wandb.init(project=code_config.TASKB_SUMMARY_WANDB_PROJECT, \
               name=model_name, \
               save_code=True, \
               job_type=code_config.TASKB_SUMMARY_JOB_TYPE, \
               notes=code_config.TASKB_SUMMARY_NOTES)

    cfg = wandb.config
    cfg.update({
        "epochs": code_config.TASKB_SUMMARY_EPOCHS, \
        "batch_size": code_config.TASKB_SUMMARY_BATCH_SIZE, \
        "training_samples": len(train_dataloader.dataset), \
        "validation_samples": len(valid_dataloader.dataset), \
        "lr": code_config.TASKB_SUMMARY_LEARNING_RATE, \
        "gradient_accumulation_steps": code_config.TASKB_GRADIENT_ACCUMULATION_STEPS, \
        "scheduler": scheduler_name, \
        "max_source_length": code_config.TASKB_SUMMARY_MAX_SOURCE_LENGTH, \
        "min_target_length": summary_min_target_length, \
        "max_target_length": summary_max_target_length
    })

    training_loop(model_name, model, \
                  train_dataloader, \
                  valid_dataloader, \
                  optimizer, lr_scheduler)


# %%
kfold = KFold(n_splits=code_config.TASKB_SUMMARY_N_SPLITS,shuffle=True,random_state=code_config.SEED)
for section, section_df in df_by_section.items():
    text_column = df_by_section[section].columns[0]
    summary_column = df_by_section[section].columns[1]
    for split,(train_idx,valid_idx) in enumerate(kfold.split(section_df)):
        final_train_loop(section_df, \
                         train_idx, valid_idx, \
                         section, split, \
                         tokenizer, text_column, summary_column, \
                         min_length_by_section[section], \
                         max_length_by_section[section])

