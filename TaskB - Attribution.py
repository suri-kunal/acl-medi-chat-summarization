#!/usr/bin/env python
# coding: utf-8
# %%
import pandas as pd
import plotly
import plotly.express as px
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from pathlib import Path
from datasets import Dataset,DatasetDict,load_dataset,load_metric
import evaluate
import re
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
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
import deepdiff


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
ML_TOKENIZER = "emilyalsentzer/Bio_ClinicalBERT"
problem_type = "multi_label_classification"


# %%
model_dict = dict()
threshold_dict = dict()
config = AutoConfig.from_pretrained(ML_TOKENIZER, \
                                    force_download=True)
config.num_labels = 20
config.problem_type = "multi_label_classification"
tokenizer = AutoTokenizer.from_pretrained(ML_TOKENIZER, \
                                          do_lower_case=True, \
                                          force_download=True)
for split in [0,1,2,3,4]:
    ML_CHECKPOINT = \
    f"suryakiran786/bio-clinicalbert-multilabel-focal-loss-seed-42-complete-data-{split}-roc-pr"
    THRESHOLD_FILE = f"threshold-{split}.json"
    model = AutoModelForSequenceClassification.from_pretrained(ML_CHECKPOINT, \
                                                               config=config, \
                                                               force_download=True)
    
    with open(THRESHOLD_FILE,"r") as f:
        threshold = json.load(f)
        
    model_dict[split] = model
    threshold_dict[split] = threshold


# %%
train_path = Path.cwd().joinpath("mediqa-chat-data","TaskB","TaskB-TrainingSet.csv")
validation_path = Path.cwd().joinpath("mediqa-chat-data","TaskB","TaskB-ValidationSet.csv")

train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(validation_path)
merge_df = pd.concat([train_df,valid_df],axis=0,ignore_index=True)
#merge_df = merge_df.sample(5)


# %%
def predict_forward_func(input_ids, token_type_ids=None, 
                         position_ids=None, attention_mask=None):
    """Function passed to ig constructors"""
    return model(input_ids=input_ids, 
                 token_type_ids=token_type_ids, 
                 position_ids=position_ids, 
                 attention_mask=attention_mask)[0]  


def prepare_input(text):
    """Prepare ig attribution input: tokenize sample and baseline text."""
    tokenized_text = tokenizer(text, \
                               return_tensors="pt", \
                               padding="max_length", \
                               max_length = code_config.MULTI_LABEL_ATTRIBUTION_LENGTH, \
                               truncation=True, \
                               return_attention_mask=True)
    seq_len = tokenized_text["input_ids"].shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    # Construct the baseline (a reference sample).
    # A sequence of [PAD] tokens of length equal to that of the processed sample
    ref_text = tokenizer.pad_token * (seq_len - 2) # special tokens
    tokenized_ref_text = tokenizer(ref_text, return_tensors="pt") 
    ref_position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    return (tokenized_text["input_ids"],
            tokenized_text["token_type_ids"], 
            position_ids,
            tokenized_ref_text["input_ids"],
            tokenized_ref_text["token_type_ids"], 
            ref_position_ids,
            tokenized_text["attention_mask"])

def place_on_device(*tensors):
    tensors_device = []
    for t in tensors:
        tensors_device.append(t.to(device))
    return tuple(tensors_device)  

def lig_attribute(lig, class_index, input_data):
    return lig.attribute(
        inputs=input_data[0], \
        baselines=input_data[3],
        additional_forward_args=(input_data[1], input_data[2], input_data[6]), \
        return_convergence_delta=True, \
        target=class_index, \
        n_steps=200)


# %%
def get_preds(sentence,model,threshold):
    
    threshold_dict = threshold

    model = model.to(device)
    model.eval()
    
    # Input for lig attributions (model with no special layers configured)
    input_data = place_on_device(*prepare_input(sentence))
    
    with torch.no_grad():
        input_ids = input_data[0]
        token_type_ids = input_data[1]
        attention_mask = input_data[-1]
        
        output = model(input_ids=input_ids, \
                       token_type_ids=token_type_ids, \
                       attention_mask=attention_mask, \
                       return_dict=True)
    logits = output.logits.detach().cpu().squeeze()
    
    predicted_idx = []
    for idx,(section,threshold) in enumerate(threshold_dict.items()):
        if logits[idx] > threshold:
            predicted_idx.append(idx)
    
    return predicted_idx


# %%
def get_word_embedding_attribution(sentence,idx,model):
    """
    Getting layer level attributions for given sentence
    Shape -> (batch, max_length, embedding_dimensions)
    """
    model = model.to(device)
    model.eval()
    
    # Input for lig attributions (model with no special layers configured)
    input_data = place_on_device(*prepare_input(sentence))
    
    def predict_forward_func(input_ids, token_type_ids=None, 
                         position_ids=None, attention_mask=None):
        """Function passed to ig constructors"""
        return model(input_ids=input_ids, 
                     token_type_ids=token_type_ids, 
                     position_ids=position_ids, 
                     attention_mask=attention_mask)[0]  

    # 1. Layer: model.bert.embeddings.word_embeddings
    lig_we = LayerIntegratedGradients(
        predict_forward_func, 
        model.bert.embeddings.word_embeddings)
    layer_attributions_we, _ = lig_attribute(lig_we, idx, input_data)
    
    return layer_attributions_we


# %%
def reorder_encounter_data(encounter_data_dict):
    """
    Reorder encounter_data to encounter_id, section_id, utterance_id,utterance,attribution
    """
    encounter_data_ranked = dict()
    for idx,(encounter_id,encounter_data) in enumerate(encounter_data_dict.items()):
        encounter_data_ranked[encounter_id] = dict()
        for utterance_id,utterance_data in encounter_data.items():
            utterance = utterance_data[0]
            section_dict = utterance_data[1]
            for section_id, attribution in section_dict.items():
                if section_id not in encounter_data_ranked[encounter_id]:
                    encounter_data_ranked[encounter_id][section_id] = []
                encounter_data_ranked[encounter_id][section_id].append([utterance_id,utterance,attribution])
    return encounter_data_ranked


# %%
section_dict = dict()
for idx,row in merge_df.iterrows():
    section_dict[row["encounter_id"]] = dict()
    dialogue_list = row["dialogue"].split("\n")
    for utterance_idx,utterance in enumerate(dialogue_list):
        section_dict[row["encounter_id"]][utterance_idx] = dict()
        utterance_dict = dict()
        for split,model in model_dict.items():
            preds = get_preds(utterance,model,threshold_dict[split])
            if isinstance(preds,int):
                preds = [preds]
            for pred in preds:
                if pred not in utterance_dict:
                    utterance_dict[pred] = [split]
                else:
                    utterance_dict[pred].append(split)
            section_dict[row["encounter_id"]][utterance_idx] = (utterance,utterance_dict)


# %%
# Get attribution for every prediction
if Path("encounter_data_with_attribution.json").exists():
    with open("encounter_data_with_attribution.json","r") as f:
        encounter_data_ranked = json.load(f)
else:
    encounter_data_ranked = dict()
encounter_data_dict = dict()
for encounter_id,encounter_body in tqdm(section_dict.items()):
    if encounter_id in encounter_data_ranked:
        print(f"Skipping {encounter_id}")
        continue
    encounter_data_dict[encounter_id] = dict()
    for utterance_id,utterance_body in encounter_body.items():
        utterance = utterance_body[0]
        section_attribution_dict = dict()
        for section_id,split_list in utterance_body[1].items():
            attribute_by_split = []
            for split in split_list:
                attribution = get_word_embedding_attribution(utterance,section_id,model_dict[split])
                attribution = attribution.mean(dim=-1)
                attribution = attribution.abs().mean(dim=-1).squeeze(0)
                attribute_by_split.append(attribution.item())
            section_attribution_dict[str(section_id)] = np.mean(attribute_by_split).item()
        encounter_data_dict[str(encounter_id)][str(utterance_id)] = (utterance,section_attribution_dict)
    reordered_encounter_data = reorder_encounter_data(encounter_data_dict)
    encounter_data_ranked.update(reordered_encounter_data)
    with open("encounter_data_with_attribution_tmp.json","w") as f:
        json.dump(encounter_data_ranked,f,indent=2)
    with open("encounter_data_with_attribution_tmp.json","r") as f:
        encounter_data_ranked_2 = json.load(f)
    if deepdiff.diff.DeepDiff(encounter_data_ranked,encounter_data_ranked_2) != {}:
        raise Exception("encounter_data_ranked must be equal to encounter_data_ranked_2")
    else:
        with open("encounter_data_with_attribution.json","w") as f:
            json.dump(encounter_data_ranked,f,indent=2)
    print(f"{encounter_id} done!")

