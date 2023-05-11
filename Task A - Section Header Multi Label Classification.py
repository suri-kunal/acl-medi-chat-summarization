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


# %%
os.environ["WANDB_API_KEY"] = code_config.WANDB_API
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"
hf_hub.login(code_config.HF_API,add_to_git_credential=True)
WANDB_PROJECT = code_config.MULTI_LABEL_WANDB_PROJECT


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
tokenizer = AutoTokenizer.from_pretrained(code_config.MULTI_LABEL_MODEL_CHECKPOINT, \
                                          do_lower_case=True, \
                                          force_download=True)


# %%
def my_tokenizer(data,labels,max_length):
    
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
def log_validation_predictions(full_input_ids, input_ids, labels, logits):
    
    if len(full_input_ids) != len(input_ids):
        raise Exception("Length of full_input_ids must be equal to length of truncated_input_ids")
    
    if len(input_ids) != len(labels):
        raise Exception("Length of truncated_input_ids must be equal to length of labels")
    
    if len(labels) != len(logits):
        raise Exception("Length of labels must be equal to length of logits")

    columns = ["id","full_sentence","truncated_sentence"]
    for section in label_df.columns:
        columns.append(f"{section}_Label")
        columns.append(f"{section}_Predicted")
    for section in label_df.columns:
        columns.append(f"{section}_Scores")
    valid_table = wandb.Table(columns=columns)
        
    full_input_ids = torch.cat(full_input_ids,dim=0)
    input_ids = torch.cat(input_ids,dim=0)
    label_tensor = torch.cat(labels,dim=0).float()
    logit_tensor = torch.cat(logits,dim=0).float() 
    
    all_tokens = full_input_ids
    tokens = input_ids
    scores = F.sigmoid(logit_tensor)
    log_full_input_ids = full_input_ids
    log_truncated_input_ids = input_ids
    log_labels = label_tensor.detach().cpu()
    log_scores = scores.detach().cpu()
    log_preds = (log_scores > 0.5).to(torch.uint8)
    
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
    total_train_loss = 0

    model.train()
    if model.training is False:
        raise Exception("Model must be trainable")

    for train_step, batch in enumerate(train_dl):

        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_full_sentences = batch[-1]

        model.zero_grad()
        
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
            roc_auc_score(labels_epoch_tensor[:,idx],preds_epoch_tensor[:,idx])
        except Exception as e:
            roc_score = 0
        try:
            pr_score = \
            average_precision_score(labels_epoch_tensor[:,idx],preds_epoch_tensor[:,idx])
        except Exception as e:
            pr_score = 0
        roc_dict[f"ROC/{section}"] = roc_score
        pr_dict[f"PR/{section}"] = pr_score

    avg_eval_loss = total_eval_loss / len(valid_dl)

    if only_inference is False:
        return full_input_ids, truncated_input_ids, \
                labels_list, logits_list, \
                avg_eval_loss, \
                roc_dict, pr_dict, \
                model
    return full_input_ids, truncated_input_ids, \
            labels_list, logits_list, \
            avg_eval_loss, \
            roc_dict, pr_dict, \
            None


# %%
def train_valid_fn(model_checkpoint,job_type,notes,num_classes,train_dl,valid_dl,epochs,lr,split):
    
    num_labels = num_classes
    problem_type = "multi_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, \
                                                               num_labels=num_labels, \
                                                               problem_type=problem_type, \
                                                               force_download=True)
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(),lr=lr,eps=1e-8)
    
    total_steps = len(train_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, \
                                                num_warmup_steps=0, \
                                                num_training_steps=total_steps)
    
    model_name = model_checkpoint.split("/")[-1]
    model_name = f"{model_name}-{problem_type}-{epochs}-{num_labels}-{split}-roc-pr"
    wandb.init(project=code_config.MULTI_LABEL_WANDB_PROJECT, \
               name=model_name, \
               save_code=True, \
               job_type=job_type, \
               resume=None, \
               notes=notes)
    cfg = wandb.config
    cfg.update({"epochs":epochs, \
                "batch_size":train_dl.batch_size, \
                "lr":lr, \
                "training_samples":len(train_dl.dataset) \
                ,"validation_samples":len(valid_dl.dataset)})

    best_loss = np.inf
    best_model = None
    best_epoch = 0
    # Training loop - only_inference is always False    
    for epoch_i in tqdm(range(epochs)):
        
        model,avg_training_loss = train_fn(model,train_dl,optimizer,scheduler,epoch_i)
        
        full_input_ids, truncated_input_ids, \
        labels_list, logits_list, \
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
    
    new_model = AutoModelForSequenceClassification.from_pretrained(f"suryakiran786/{model_name}", \
                                                               num_labels=num_labels, \
                                                               problem_type=problem_type, \
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
    for idx,section in enumerate(label_df.columns):
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
    
    log_validation_predictions(full_input_ids, truncated_input_ids, labels_list, logits_list)
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
    train_valid_fn(model_checkpoint = code_config.MULTI_LABEL_MODEL_CHECKPOINT, \
                   job_type = code_config.MULTI_LABEL_JOB_TYPE, \
                   notes = code_config.MULTI_LABEL_NOTES, \
                   num_classes = len(label_df.columns), \
                   train_dl = train_dl, \
                   valid_dl = valid_dl, \
                   epochs = code_config.MULTI_LABEL_EPOCHS, \
                   lr = code_config.MULTI_LABEL_LEARNING_RATE, \
                   split = split)


# %%

