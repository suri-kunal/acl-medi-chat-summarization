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
                         SchedulerType, \
                         AutoModelForSequenceClassification, \
                         GenerationConfig

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
import gc
from tqdm import tqdm
import warnings
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
warnings.filterwarnings("ignore")
tqdm.pandas()


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


train_path = Path.cwd().joinpath("mediqa-chat-data", "TaskA", "TaskA-TrainingSet.csv")
validation_path = Path.cwd().joinpath(
    "mediqa-chat-data", "TaskA", "TaskA-ValidationSet.csv"
)

train_df = pd.read_csv(train_path, index_col="ID")
valid_df = pd.read_csv(validation_path, index_col="ID")
merge_df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
merge_df["dialogue_wo_whitespaces"] = merge_df["dialogue"].apply(
    lambda x: re.sub(r"[\r\n\s]+", " ", x)
)
merge_df.reset_index(inplace=True)
merge_df.rename(mapper={"index": "ID"}, axis=1, inplace=True)

with open("TaskA-label2idx.json","r") as f:
    label2idx = json.load(f)
    
with open("TaskA-idx2label.json","r") as f:
    idx2label = json.load(f)

merge_df["label"] = merge_df["section_header"].apply(lambda x: label2idx[x])


# %%


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# %%


def postprocess_text(preds,labels):
    seed_everything(code_config.TASKA_SUMMARY_SEED)
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    
    return preds, labels


# %%


scorers = {
        'rouge': (
            evaluate.load('rouge'),
            {'use_aggregator': False},
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        ),
#         'bert_scorer': (
#             evaluate.load('bertscore'),
#             {'model_type': 'microsoft/deberta-xlarge-mnli'},
#             ['precision', 'recall', 'f1'],
#             ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
#         ),
#         'bluert': (
#             evaluate.load('bleurt', config_name='BLEURT-20'),
#             {},
#             ['scores'],
#             ['bleurt']
#         ),
    }


# %%


def calculate_metrics(references,predictions,scorer,key,save_key,**kwargs):
        scores = scorer.compute(references=references, predictions=predictions, **kwargs)
        if isinstance(scores[key],list):
            if len(scores[key]) > 1:
                raise Exception("scores[key] have more than one elements")
            return scores[key][0]
        return scores[key]


# %%
def filter_and_aggregate(obj, indices):
    agg_obj = {}
    for k, v in obj.items():
        agg_obj[k] = float(np.mean([v[i] for i in indices]))
    return agg_obj

# %%


TASKA_SUMMARY_TOKENIZER_MODEL_MAPPING = \
{
    "GanjinZero/biobart-v2-base": \
    "suryakiran786/5-fold-stratified-cv-biobart-v2-base-with-section-description-complete-data",
    "google/flan-t5-large": \
    "suryakiran786/5-fold-stratified-cv-flan-t5-large-with-section-description-complete-data", 
    "MingZhong/DialogLED-large-5120": \
    "suryakiran786/5-fold-stratified-cv-dialogled-large-with-section-description-complete-data",
    "MingZhong/DialogLED-base-16384": \
    "suryakiran786/5-fold-stratified-cv-dialogled-base-with-section-description-complete-data"
}
TASKA_MULTI_CLASS_MODEL_MAPPING = \
{
    "emilyalsentzer/Bio_ClinicalBERT": \
    "suryakiran786/5-stratified-cv-bio-clinicalbert-multiclass-focal-loss-seed-42-complete-data"
}


# %%


wandb_kwargs = {"project": "metric-optimization-hpo","group":"contrastive-search"}
wandbc = WeightsAndBiasesCallback(metric_name="rouge_bertscore_bleurt_score", \
                                  wandb_kwargs=wandb_kwargs, \
                                  as_multirun=True)


# %%


def metric_calculation(summary_tokenizer, summary_model, \
                       classifier_tokenizer, classifier_model, \
                       df, split):
    
    @wandbc.track_in_wandb()
    def objective(trial):
        
        max_length = trial.suggest_int("max_length", \
                                       code_config.TASKA_SUMMARY_MIN_TARGET_LENGTH, \
                                       code_config.TASKA_SUMMARY_MAX_TARGET_LENGTH)
        penalty_alpha = trial.suggest_float("penalty_alpha",0.1,10.)
        top_k = trial.suggest_int("top_k",1,15)
        
        generate_kwargs = {
            "min_length": code_config.TASKA_SUMMARY_MIN_TARGET_LENGTH, \
            "max_length": max_length, \
            "penalty_alpha": penalty_alpha, \
            "top_k": top_k
        }
    
        TASKA_SUMMARY_TOKENIZER = \
        f"{summary_tokenizer}"
        TASKA_SUMMARY_CHECKPOINT = \
        f"{summary_model}-{split}"

        MULTI_CLASS_TOKENIZER = f"{classifier_tokenizer}"
        MULTI_CLASS_CHECKPOINT = f"{classifier_model}-{split}"

        multi_class_config = AutoConfig.from_pretrained(MULTI_CLASS_CHECKPOINT)
        multi_class_config.num_labels = 20

        taska_summary_config = AutoConfig.from_pretrained(TASKA_SUMMARY_CHECKPOINT)

        multi_class_tokenizer = AutoTokenizer.from_pretrained(MULTI_CLASS_TOKENIZER, \
                                                                do_lower_case=True, \
                                                                force_download=True)

        taska_summary_tokenizer = AutoTokenizer.from_pretrained(TASKA_SUMMARY_TOKENIZER, \
                                                                do_lower_case=True, \
                                                                force_download=True)

        multi_class_model = \
        AutoModelForSequenceClassification.from_pretrained(MULTI_CLASS_CHECKPOINT, \
                                                           config=multi_class_config, \
                                                           force_download=True)

        taska_summary_model = \
        AutoModelForSeq2SeqLM.from_pretrained(TASKA_SUMMARY_CHECKPOINT, \
                                              config=taska_summary_config, \
                                              force_download=True)

        multi_class_model = multi_class_model.to(device)
        multi_class_model.eval()

        taska_summary_model = taska_summary_model.to(device)
        taska_summary_model.eval()

        test_df = df
        test_df["predicted_section_header"] = None
        test_df["predicted_section_text_postprocessed"] = None
        test_df["reference_section_text_postprocessed"] = None

        for idx in test_df.index:
            sentence = test_df.loc[idx,"dialogue_wo_whitespaces"]

            tokenized_sentence = \
            multi_class_tokenizer.encode_plus(sentence,
                                            add_special_tokens=True,
                                            padding="max_length",
                                            truncation=True,
                                            max_length=code_config.MULTI_CLASS_MAX_LENGTH,
                                            verbose=False,
                                            return_tensors="pt",
                                            return_attention_mask=True)

            input_ids = tokenized_sentence["input_ids"].to(device)
            token_type_ids = tokenized_sentence["token_type_ids"].to(device)
            attention_mask = tokenized_sentence["attention_mask"].to(device)

            with torch.no_grad():
                preds = multi_class_model(input_ids=input_ids, \
                                          token_type_ids=token_type_ids, \
                                          attention_mask=attention_mask)

                preds = preds.logits.detach().cpu().numpy().squeeze(0)

                best_idx = np.argmax(preds)
                section_header = idx2label[str(best_idx)]
                test_df.loc[idx,"predicted_section_header"] = section_header

        test_df["predicted_section_header_desription"] = \
        test_df["predicted_section_header"].apply(lambda x: " and ".join(section_header_mapping[x.lower()]))
        test_df["predicted_section_header_desription"] = \
        test_df["predicted_section_header_desription"].str.lower()

        summary_column = "section_text"
        text_column = "dialogue_w_section_header_desc"
        test_df[text_column] = \
        test_df["predicted_section_header_desription"] + \
        f" {str(taska_summary_tokenizer.sep_token)} " + \
        test_df["dialogue_wo_whitespaces"]

        for idx in tqdm(test_df.index):
            sentence = test_df.loc[idx,text_column]
            summary = test_df.loc[idx,summary_column]        

            model_inputs = \
            taska_summary_tokenizer(sentence, \
                                    padding=code_config.TASKA_SUMMARY_PADDING, \
                                    truncation=True, \
                                    max_length=code_config.TASKA_SUMMARY_MAX_SOURCE_LENGTH, \
                                    return_tensors="pt")

            labels = \
            taska_summary_tokenizer(text_target=summary, \
                                    padding=code_config.TASKA_SUMMARY_PADDING, \
                                    truncation=True, \
                                    max_length=code_config.TASKA_SUMMARY_MAX_TARGET_LENGTH, \
                                    return_tensors="pt")

            model_inputs["labels"] = labels["input_ids"]

            with torch.no_grad():

                input_ids = model_inputs["input_ids"].to(device)
                attention_mask = model_inputs["attention_mask"].to(device)
                labels = model_inputs["labels"].to(device)

                generated_tokens = \
                taska_summary_model.generate(inputs=input_ids, \
                                             attention_mask=attention_mask, \
                                             **generate_kwargs)

                if isinstance(generated_tokens,tuple):
                    generated_tokens = generated_tokens[0]

                generated_tokens_decoded = \
                taska_summary_tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
                labels_w_padding_tokens = \
                [[l.item() if l != -100 else taska_summary_tokenizer.pad_token_id for l in label] \
                 for label in labels.cpu()]
                labels_decoded = \
                taska_summary_tokenizer.batch_decode(labels_w_padding_tokens,skip_special_tokens=True)

                generated_tokens_decoded,labels_decoded = \
                postprocess_text(generated_tokens_decoded,labels_decoded)

                test_df.loc[idx,"predicted_section_text_postprocessed"] = \
                generated_tokens_decoded[0]

                test_df.loc[idx,"reference_section_text_postprocessed"] = \
                labels_decoded[0]

                
        references = test_df['reference_section_text_postprocessed'].tolist()
        predictions = test_df['predicted_section_text_postprocessed'].tolist()
        test_df['dataset'] = 0
        num_test = len(test_df)
        
        all_scores = {}
        for name, (scorer, kwargs, keys, save_keys) in scorers.items():
            print(name)
            scores = scorer.compute(references=references, predictions=predictions, **kwargs)
            for score_key, save_key in zip(keys, save_keys):
                all_scores[save_key] = scores[score_key]
                
        cohorts = [
        ('all', list(range(num_test))),
        ]
        
        outputs = {k: filter_and_aggregate(all_scores, idxs) for (k, idxs) in cohorts}
        
        rouge1 = outputs["all"]["rouge1"]
        rouge2 = outputs["all"]["rouge2"]
        rougeL = outputs["all"]["rougeL"]
        rougeLsum = outputs["all"]["rougeLsum"]
#         bertscore_f1 = outputs["all"]["bertscore_f1"]
#         bleurt = outputs["all"]["bleurt"]
        
        return rouge2
        
        
    
    return objective


# %%


skf = StratifiedKFold(n_splits=code_config.TASKA_SUMMARY_N_SPLITS,shuffle=True,random_state=code_config.SEED)
for split,(train_idx,valid_idx) in enumerate(skf.split(merge_df, y=merge_df["label"])):
    for summary_tokenizer, summary_model in TASKA_SUMMARY_TOKENIZER_MODEL_MAPPING.items():
        for mc_tokenizer,mc_model in TASKA_MULTI_CLASS_MODEL_MAPPING.items():
            
            test_df = merge_df.iloc[valid_idx]
            objective_fn = \
            metric_calculation(summary_tokenizer, \
                   summary_model, \
                   mc_tokenizer, 
                   mc_model, \
                   test_df, split)
            
            study_name = \
            "contrastive-search" + \
            "-" + \
            summary_tokenizer.split("/")[-1] + \
            "-" + \
            mc_tokenizer.split("/")[-1] + \
            "-" + \
            str(split)
            
            study = optuna.create_study(study_name=study_name, \
                                        direction="maximize")
            study.optimize(objective_fn, n_trials=5, callbacks=[wandbc])

