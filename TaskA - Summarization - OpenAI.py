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
import time

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
from sentence_transformers import SentenceTransformer, util
import openai
import sys
warnings.filterwarnings("ignore")
tqdm.pandas()


# %%
os.environ["WANDB_API_KEY"] = code_config.WANDB_API
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "online"
hf_hub.login(code_config.HF_API)
os.environ["OPENAI_API_KEY"] = code_config.OPENAI_API
openai.api_key = os.getenv("OPENAI_API_KEY")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

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
        'bert_scorer': (
            evaluate.load('bertscore'),
            {'model_type': 'microsoft/deberta-xlarge-mnli',"batch_size":4},
            ['precision', 'recall', 'f1'],
            ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
        ),
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
TASKA_MULTI_CLASS_MODEL_MAPPING = \
{
    "emilyalsentzer/Bio_ClinicalBERT": \
    "suryakiran786/5-stratified-cv-bio-clinicalbert-multiclass-focal-loss-seed-42-complete-data"
}


# %%
def openai_complete(text,max_length):
    result = openai.Completion.create(
              model="text-davinci-003",
              prompt=text,
              temperature=0.,
              max_tokens=max_length,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0
            )
    return result["choices"][0]["text"]


# %%
def metric_calculation(classifier_tokenizer, classifier_model, \
                       train_df, test_df, \
                       split, no_of_iterations, no_of_samples):
    
    MULTI_CLASS_TOKENIZER = f"{classifier_tokenizer}"
    MULTI_CLASS_CHECKPOINT = f"{classifier_model}"

    multi_class_config = AutoConfig.from_pretrained(MULTI_CLASS_CHECKPOINT)
    multi_class_config.num_labels = 20

    multi_class_tokenizer = AutoTokenizer.from_pretrained(MULTI_CLASS_TOKENIZER, \
                                                            do_lower_case=True, \
                                                            force_download=False)

    multi_class_model = \
    AutoModelForSequenceClassification.from_pretrained(MULTI_CLASS_CHECKPOINT, \
                                                       config=multi_class_config, \
                                                       force_download=False)

    multi_class_model = multi_class_model.to(device)
    multi_class_model.eval()

    wandb.init(project="openai-summarization", \
               name=f"openai-summarization-with-{no_of_iterations}-iterations-{no_of_samples}-samples-{split}-split", \
               save_code=True)
    
    train_df = train_df
    test_df = test_df
    test_df["predicted_section_header"] = None
    test_df["predicted_summary"] = None
    test_df["best_idx"] = None
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

    for idx in tqdm(test_df.index):
        test_sentence = test_df.loc[idx,"dialogue_wo_whitespaces"]
        test_section = test_df.loc[idx,"predicted_section_header"].upper()
        test_summary = test_df.loc[idx,"section_text"]
        
        predicted_summary_list = []
        for _ in range(no_of_iterations):
            train_subset = train_df.loc[train_df["section_header"] == test_section]
            train_sample_indices = \
            np.random.choice(train_subset.index,size=min(no_of_samples,train_subset.shape[0]),replace=False)
            if not isinstance(train_sample_indices,list):
                train_sample_indices = [train_sample_indices]
            
            prompts_list = []
            for train_idx in train_sample_indices:
                train_dialog = train_subset.loc[train_idx,"dialogue_wo_whitespaces"]
                train_summary = train_subset.loc[train_idx,"section_text"]
                prompts_list.append(f"{train_dialog} -> {train_summary}")
            prompts_list.append(f"{test_sentence} ->")
            final_prompt = "\n".join(prompts_list)
            predicted_summary = \
            openai_complete(final_prompt,code_config.TASKA_SUMMARY_MAX_TARGET_LENGTH)
            predicted_summary_list.append(predicted_summary)
            
        best_summary = None
        best_idx = None
        embedding_dict = dict()
        if len(predicted_summary_list) < 1:
            raise Exception("predicted_summary_list must contain atleast one element")
        if len(predicted_summary_list) != no_of_iterations:
            raise Exception("predicted_summary_list must be equal to no_of_iterations")
        if len(predicted_summary_list) == 1:
            best_summary = predicted_summary_list[0]
            best_idx = 0
        else:
            for summary_idx,summary in enumerate(predicted_summary_list):
                embeddings = sentence_model.encode(summary,convert_to_tensor=True)
                embedding_dict[summary_idx] = embeddings.cpu().numpy()

            similarity_dict = {}
            for primary_idx, primary_embedding in embedding_dict.items():
                similarity_list = []
                for secondary_idx, secondary_embedding in embedding_dict.items():
                    if primary_idx != secondary_idx:
                        cosine_sim = util.cos_sim(primary_embedding,secondary_embedding).item()
                        similarity_list.append(cosine_sim)
                avg_cosine_sim = np.mean(similarity_list)
                similarity_dict[primary_idx] = avg_cosine_sim

            idx_list,similarity_list = zip(*similarity_dict.items())

            best_similarity_index = np.argmax(similarity_list)
            best_idx = idx_list[best_similarity_index]
            best_summary = predicted_summary_list[best_idx]

        test_df.loc[idx,"predicted_summary"] = best_summary.strip()
        test_df.loc[idx,"best_idx"] = best_idx

        generated_tokens_decoded,labels_decoded = \
        postprocess_text([best_summary],[test_summary])

        test_df.loc[idx,"predicted_section_text_postprocessed"] = \
        generated_tokens_decoded[0]

        test_df.loc[idx,"reference_section_text_postprocessed"] = \
        labels_decoded[0]
            
    references = test_df['reference_section_text_postprocessed'].tolist()
    predictions = test_df['predicted_section_text_postprocessed'].tolist()
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
    bertscore_f1 = outputs["all"]["bertscore_f1"]

    cfg = wandb.config
    cfg.update({
        "Rouge1": rouge1, \
        "Rouge2": rouge2, \
        "RougeL": rougeL, \
        "RougeLsum": rougeLsum, \
        "bertscore_f1": bertscore_f1
    })
    wandb.finish()

# %%
skf = StratifiedKFold(n_splits=code_config.TASKA_SUMMARY_N_SPLITS,shuffle=True,random_state=code_config.SEED)
for split,(train_idx,valid_idx) in enumerate(skf.split(merge_df, y=merge_df["label"])):     
    train_df = merge_df.iloc[train_idx]
    valid_df = merge_df.iloc[valid_idx]
    [*mc_tokenizer],[*mc_model] = zip(*TASKA_MULTI_CLASS_MODEL_MAPPING.items())
    mc_tokenizer = mc_tokenizer[0]
    mc_model = mc_model[0]
    mc_model = f"{mc_model}-{split}"
    for no_of_iterations in [1]:
        for no_of_samples in [9,7,5,3,1]:
            metric_calculation(mc_tokenizer, mc_model, \
                               train_df, valid_df, \
                               split, no_of_iterations, no_of_samples)
    break
# %%
