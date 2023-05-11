#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import pandas as pd
import plotly
import plotly.express as px
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from pathlib import Path
import datasets as ds # Dataset,DatasetDict,load_dataset,load_metric
import evaluate
import re
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
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
from collections import OrderedDict
from transformer_classifier import BertForSequenceClassificationML, \
                                   DebertaV2ForSequenceClassificationML
import json


# %%
os.environ["WANDB_API_KEY"] = code_config.WANDB_API
os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_hub.login(code_config.HF_API,add_to_git_credential=True)
WANDB_PROJECT = code_config.MULTI_LABEL_WANDB_PROJECT


# %%
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# %%
train_path = Path.cwd().joinpath("mediqa-chat-data","TaskA","TaskA-TrainingSet.csv")
validation_path = Path.cwd().joinpath("mediqa-chat-data","TaskA","TaskA-ValidationSet.csv")

train_df = pd.read_csv(train_path,index_col="ID")
valid_df = pd.read_csv(validation_path,index_col="ID")
merge_df = pd.concat([train_df,valid_df],axis=0,ignore_index=True)
merge_df["dialogue_wo_whitespaces"] = merge_df["dialogue"].apply(lambda x: re.sub(r'[\r\n\s]+',' ',x))
merge_df.reset_index(inplace=True)
merge_df.rename(mapper={'index':'ID'},axis=1,inplace=True)
if code_config.MULTI_LABEL_SAMPLING is True:
    merge_df = merge_df.sample(50)
section_header = merge_df.pop("section_header")
label_df = pd.get_dummies(section_header)


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
config = AutoConfig.from_pretrained(code_config.MULTI_LABEL_MODEL_CHECKPOINT, \
                         force_download=True)
config.balanced_loss = code_config.MUTLI_LABEL_BALANCE_LOSS
tokenizer = AutoTokenizer.from_pretrained(code_config.MULTI_LABEL_MODEL_CHECKPOINT, \
                                          do_lower_case=True, \
                                          force_download=True)


# %%
def my_tokenizer(data,labels,max_length):

    seed_everything(code_config.MULTI_LABEL_SEED)
    
    complete_input_ids = []
    input_ids = []
    attention_mask = []
    
    for sentence in data:
        non_truncated_sentence = \
        tokenizer.encode(sentence, \
                         return_tensors="pt", \
                         padding="max_length", \
                         truncation=True, \
                         verbose=False, \
                         max_length=3000)
        complete_input_ids.append(non_truncated_sentence)
        
        tokenized_sentence = \
        tokenizer.encode_plus(sentence, \
                              add_special_tokens=True, \
                              padding="max_length", \
                              truncation=True, \
                              max_length=code_config.MULTI_LABEL_MAX_LENGTH, \
                              verbose=False, \
                              return_tensors="pt", \
                              return_attention_mask=True)
        input_ids.append(tokenized_sentence["input_ids"])
        attention_mask.append(tokenized_sentence["attention_mask"])
    
    non_truncated_sentence_tensors = torch.cat(complete_input_ids,dim=0)
    input_ids_tensor = torch.cat(input_ids,dim=0)
    attention_mask_tensor = torch.cat(attention_mask,dim=0)
    labels_tensor = torch.tensor(labels.to_numpy().astype("float"))
    
    return input_ids_tensor,attention_mask_tensor,labels_tensor,non_truncated_sentence_tensors


# %%
def create_dataset(input_ids_tensor,attention_mask_tensor,labels_tensor,full_sentence):
    return TensorDataset(input_ids_tensor,attention_mask_tensor,labels_tensor,full_sentence)

def create_dataloader(dataset,sampler,batch_size,num_workers):
    return DataLoader(dataset,sampler=sampler,batch_size=batch_size,num_workers=num_workers,pin_memory=True)


# %%
def threshold_calculation(y_true,y_pred):
    precision,recall,threshold = precision_recall_curve(y_true,y_pred)    
    f1 = 2*precision*recall/(precision+recall+1e-8)
    f1 = f1[:-1]
    f1_argmax = np.argmax(f1)
    best_threshold = threshold[f1_argmax]
    return best_threshold.item()


# %%
def log_validation_predictions(full_input_ids, input_ids, labels, logits, threshold_dict):
    
    if len(full_input_ids) != len(input_ids):
        raise Exception("Length of full_input_ids must be equal to length of truncated_input_ids")
    
    if len(input_ids) != len(labels):
        raise Exception("Length of truncated_input_ids must be equal to length of labels")
    
    if len(labels) != len(logits):
        raise Exception("Length of labels must be equal to length of logits")

    columns = ["id","full_sentence","truncated_sentence"]
    for section in label_df.columns:
        columns.append(f"{section}_Label")
    for section in label_df.columns:
        columns.append(f"{section}_Predicted")
    for section in label_df.columns:
        columns.append(f"{section}_Scores")
    valid_table = wandb.Table(columns=columns)
        
    full_input_ids = torch.cat(full_input_ids,dim=0)
    input_ids = torch.cat(input_ids,dim=0)
    label_tensor = torch.cat(labels,dim=0).float()
    logit_tensor = torch.cat(logits,dim=0).float() 
    threshold_values = np.array(list(threshold_dict.values()))[None,:]
    
    if threshold_values.shape[-1] != logit_tensor.shape[-1]:
        raise Exception("Entities in threshold are not equal to predicted entities")
    
    all_tokens = full_input_ids
    tokens = input_ids
    scores = F.sigmoid(logit_tensor)
    log_full_input_ids = full_input_ids
    log_truncated_input_ids = input_ids
    log_labels = label_tensor.detach().cpu().numpy()
    log_scores = logit_tensor.detach().cpu().numpy()
    log_preds = (log_scores >= threshold_values).astype(np.uint8)
    
    for idx,(lfs,lts,ll,lp,ls) in enumerate(zip(log_full_input_ids, \
                                                log_truncated_input_ids, \
                                                log_labels, \
                                                log_preds, \
                                                log_scores)):
        
        log_full_sentences = tokenizer.decode(lfs,skip_special_tokens=True)
        log_truncated_sentences = tokenizer.decode(lts,skip_special_tokens=True)
        list_of_y = ll.tolist() + lp.tolist() + ls.tolist()
#         assert len(list_of_y) == 60, len(list_of_y)
        sentence_id = str(idx)
        valid_table.add_data(sentence_id, log_full_sentences ,log_truncated_sentences ,*list_of_y)
        
    wandb.log({"Validation Table":valid_table})


# %%
def train_fn(model,train_dl,optimizer,scheduler,epoch):
    
    seed_everything(code_config.MULTI_LABEL_SEED)
    
    total_train_loss = 0

    model.train()
    if model.training is False:
        raise Exception("Model must be trainable")

    for train_step, batch in enumerate(train_dl):

        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_full_sentences = batch[-1]

        model.zero_grad(set_to_none=True)
        
        result = model(b_input_ids, \
                       token_type_ids = None, \
                       attention_mask = b_attention_mask, \
                       labels = b_labels, \
                       return_dict = True)

        loss = result.loss
        train_step_new = train_step + epoch*len(train_dl)        
        wandb.log({"Batch/Training Step":train_step_new + 1,"Batch/Training Loss":loss.item()})

        total_train_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

        optimizer.step()
        scheduler.step()

    avg_training_loss = total_train_loss/len(train_dl)

    return model, avg_training_loss


# %%
def valid_fn(valid_dl,model,epoch=0,only_inference=False):
    
    seed_everything(code_config.MULTI_LABEL_SEED)

    model = model.to(device)
    model.eval()
    if model.training is True:
        raise Exception("Model should not be trainable")

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    logits_list = []
    labels_list = []
    full_input_ids = []
    truncated_input_ids = []
    
    for val_step,batch in enumerate(valid_dl):        
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_full_sentences = batch[-1]

        truncated_input_ids.append(batch[0])
        full_input_ids.append(batch[-1])
        
        with torch.no_grad():
            result = model(input_ids=b_input_ids, \
                           token_type_ids=None, \
                           attention_mask=b_attention_mask, \
                           labels=b_labels, \
                           return_dict=True)

            loss = result.loss
            logits = result.logits

        if only_inference is False:
            val_step_new = val_step + epoch*len(valid_dl)        
            wandb.log({"Batch/Validation Step":val_step_new + 1,"Batch/Validation Loss":loss.item()})

        total_eval_loss += loss.item()
        logits = logits.detach().cpu()
        label_ids = b_labels.detach().cpu()

        logits_list.append(logits)
        labels_list.append(label_ids)

    logits_epoch_tensor = torch.cat(logits_list,dim=0).numpy()
    labels_epoch_tensor = torch.cat(labels_list,dim=0).numpy()
        
    roc_dict = {}
    pr_dict = {}
    for idx,section in enumerate(label_df.columns):
        try:
            roc_score = \
            roc_auc_score(labels_epoch_tensor[:,idx],logits_epoch_tensor[:,idx])
        except Exception as e:
            roc_score = 0
        try:
            pr_score = \
            average_precision_score(labels_epoch_tensor[:,idx],logits_epoch_tensor[:,idx])
        except Exception as e:
            pr_score = 0
        roc_dict[f"ROC/{section}"] = roc_score
        pr_dict[f"PR/{section}"] = pr_score
    
    avg_eval_loss = total_eval_loss / len(valid_dl)

    if only_inference is False:
        return full_input_ids, truncated_input_ids, \
                labels_list, logits_list, avg_eval_loss, \
                roc_dict, pr_dict, \
                model
    return full_input_ids, truncated_input_ids, \
            labels_list, logits_list, \
            avg_eval_loss, \
            roc_dict, pr_dict, \
            None


# %%
def train_valid_fn(num_classes,train_dl,valid_dl,split):

    seed_everything(code_config.MULTI_LABEL_SEED)
    
    config.num_labels = num_classes
    model = BertForSequenceClassificationML.from_pretrained(code_config.MULTI_LABEL_MODEL_CHECKPOINT, \
                                                             config=config, \
                                                             force_download=True)
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), \
                      lr=code_config.MULTI_LABEL_LEARNING_RATE, \
                      eps=code_config.MULTI_LABEL_ADAMW_EPS, \
                      weight_decay=code_config.MULTI_LABEL_ADAMW_WEIGHT_DECAY)
    
    total_steps = len(train_dl) * code_config.MULTI_LABEL_EPOCHS
    num_warmup_steps = int(code_config.MULTI_LABEL_WARM_UP_STEPS * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, \
                                                num_warmup_steps=num_warmup_steps, \
                                                num_training_steps=total_steps)
    
    model_name = code_config.MULTI_LABEL_MODEL_NAME
    model_name = f"{model_name}-{split}"
    wandb.init(project=code_config.MULTI_LABEL_WANDB_PROJECT, \
               name=model_name, \
               save_code=True, \
               job_type=code_config.MULTI_LABEL_JOB_TYPE, \
               resume=None, \
               notes=code_config.MULTI_LABEL_NOTES)
    cfg = wandb.config
    cfg.update({"epochs":code_config.MULTI_LABEL_EPOCHS, \
                "batch_size":train_dl.batch_size, \
                "lr":code_config.MULTI_LABEL_LEARNING_RATE, \
                "training_samples":len(train_dl.dataset), \
                "validation_samples":len(valid_dl.dataset), \
                "seed": code_config.MULTI_LABEL_SEED, \
                "AdamW_eps": code_config.MULTI_LABEL_ADAMW_EPS, \
                "AdamW_weight_decay": code_config.MULTI_LABEL_ADAMW_WEIGHT_DECAY, \
                "warm_up_steps":code_config.MULTI_LABEL_WARM_UP_STEPS})

    best_loss = np.inf
    best_model = None
    best_epoch = 0
    # Training loop - only_inference is always False    
    for epoch_i in tqdm(range(code_config.MULTI_LABEL_EPOCHS)):
        
        model,avg_training_loss = train_fn(model,train_dl,optimizer,scheduler,epoch_i)
        
        full_input_ids, truncated_input_ids, \
        _, _, \
        avg_eval_loss, \
        roc_dict, pr_dict, \
        best_model = \
        valid_fn(valid_dl,model,epoch_i,only_inference=False)
        
        if avg_eval_loss < best_loss:
            if best_model is None:
                raise Exception("bset_model cannot be None")
            best_loss = avg_eval_loss
            best_model = best_model
            best_epoch = epoch_i
        
        metrics_dict = \
                  {"Epoch/Epoch":epoch_i, \
                   "Epoch/Validation Loss":avg_eval_loss, \
                   "Epoch/Training Loss":avg_training_loss}
        metrics_dict.update(roc_dict)
        metrics_dict.update(pr_dict)
        wandb.log(metrics_dict)
    best_model.push_to_hub(model_name)
    best_model = best_model.cpu()
    del best_model
    
    
    new_model = BertForSequenceClassificationML.from_pretrained(f"suryakiran786/{model_name}", \
                                                               config=config, \
                                                               force_download=True)
    
    full_input_ids, truncated_input_ids, \
    labels_list, logits_list, \
    avg_validation_loss, \
    roc_dict, pr_dict, \
    best_model = \
    valid_fn(valid_dl,model,only_inference=True)
    
    logits_final_tensor = torch.cat(logits_list,dim=0).numpy()
    labels_final_tensor = torch.cat(labels_list,dim=0).numpy()
        
    roc_dict = {}
    pr_dict = {}
    threshold_dict = OrderedDict()
    for idx,section in enumerate(label_df.columns):
        threshold_dict[section] = \
        threshold_calculation(labels_final_tensor[:,idx],logits_final_tensor[:,idx])
        try:
            roc_score = \
            roc_auc_score(labels_final_tensor[:,idx],logits_final_tensor[:,idx])
        except Exception as e:
            roc_score = 0
        try:
            pr_score = \
            average_precision_score(labels_final_tensor[:,idx],logits_final_tensor[:,idx])
        except Exception as e:
            pr_score = 0
        roc_dict[f"Best ROC {section}"] = roc_score
        pr_dict[f"Best PR {section}"] = pr_score

    with open(f"threshold_{split}.json","w") as f:
        json.dump(threshold_dict,f)
        
    wandb.save(f"threshold_{split}.json")
    
    log_validation_predictions(full_input_ids, truncated_input_ids, labels_list, logits_list, threshold_dict)
    wandb.config.update(roc_dict)
    wandb.config.update(pr_dict)
    wandb.config.update({"Best Epoch":best_epoch + 1})
    wandb.finish()

# %%
kf = MultilabelStratifiedKFold(n_splits=code_config.MULTI_LABEL_N_SPLITS,shuffle=True,random_state=code_config.SEED)
for split,(train_idx,valid_idx) in enumerate(kf.split(merge_df,y=label_df)):
    x_train = merge_df.iloc[train_idx,:]
    y_train = label_df.iloc[train_idx,:]
    x_valid = merge_df.iloc[valid_idx,:]
    y_valid = label_df.iloc[valid_idx,:]
        
    train_input_ids_tensor,train_attention_mask_tensor,train_labels_tensor,train_full_sentences = \
    my_tokenizer(x_train["dialogue_wo_whitespaces"],y_train,code_config.MULTI_LABEL_MAX_LENGTH)
    test_input_ids_tensor,test_attention_mask_tensor,test_labels_tensor,test_full_sentences = \
    my_tokenizer(x_valid["dialogue_wo_whitespaces"],y_valid,code_config.MULTI_LABEL_MAX_LENGTH)

    train_ds = \
    create_dataset(train_input_ids_tensor,train_attention_mask_tensor,train_labels_tensor,train_full_sentences)
    valid_ds = \
    create_dataset(test_input_ids_tensor,test_attention_mask_tensor,test_labels_tensor,test_full_sentences)

    train_dl = create_dataloader(train_ds,RandomSampler(train_ds),code_config.MULTI_LABEL_BATCH_SIZE,2)
    valid_dl = create_dataloader(valid_ds,SequentialSampler(valid_ds),2*code_config.MULTI_LABEL_BATCH_SIZE,2)
    
    # model_checkpoint,job_type,notes,num_classes,train_dl,valid_dl,epochs,lr,split
    train_valid_fn(num_classes = len(label_df.columns), \
                   train_dl = train_dl, \
                   valid_dl = valid_dl, \
                   split = split)

