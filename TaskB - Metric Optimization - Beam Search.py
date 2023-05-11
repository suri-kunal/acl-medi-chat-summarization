#!/usr/bin/env python
# coding: utf-8
# %%
# Imports
from pathlib import Path
import gc

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
import time
from optuna.integration.wandb import WeightsAndBiasesCallback
from sectiontagger import SectionTagger
from itertools import chain
section_tagger = SectionTagger()
warnings.filterwarnings("ignore")
tqdm.pandas()


# %%
os.environ["WANDB_API_KEY"] = code_config.WANDB_API
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "online"
hf_hub.login(code_config.HF_API)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
model_dict = dict()
threshold_dict = dict()


MULTI_LABEL_TOKENIZER = code_config.MULTI_CLASS_MODEL_CHECKPOINT

tokenizer = AutoTokenizer.from_pretrained(MULTI_LABEL_TOKENIZER, \
                                          force_download=True)

for split in [0,1,2,3,4]:
    MULTI_LABEL_MODEL = \
    f"suryakiran786/5-fold-multilabel-cv-bio-clinicalbert-multilabel-focal-loss-seed-42-complete-data-{split}"
    THRESHOLD_FILE = f"threshold_{split}.json"

    config = AutoConfig.from_pretrained(MULTI_LABEL_MODEL)
    config.num_labels = 20
    model = AutoModelForSequenceClassification.from_pretrained(MULTI_LABEL_MODEL, \
                                                               config=config, \
                                                               force_download=True)
    model_dict[split] = model
    
    with open(THRESHOLD_FILE,"r") as f:
        threshold_dict[split] = json.load(f)
        
with open("TaskA-idx2label.json","r") as f:
    idx2label = json.load(f)

# %%
try:
    nltk.data.find("tokenizers/punkt")
except LookupError as e:
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


# %%
wandb_kwargs = {"project": "TaskB-metric-optimization-hpo-with-rouge-and-ertscore","group":"beam-search"}
wandbc = WeightsAndBiasesCallback(metric_name="rouge_bertscore_bleurt_score", \
                                  wandb_kwargs=wandb_kwargs, \
                                  as_multirun=True)

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
train_file = Path.cwd().joinpath("mediqa-chat-data","TaskB","TaskB-TrainingSet.csv")
valid_file = Path.cwd().joinpath("mediqa-chat-data","TaskB","TaskB-ValidationSet.csv")

train_df = pd.read_csv(train_file)
valid_df = pd.read_csv(valid_file)
merge_df = pd.concat([train_df,valid_df],axis=0,ignore_index=True)
# merge_df = merge_df.sample(10)

# %%
SECTION_DIVISIONS = ['subjective', 'objective_exam', 'objective_results', 'assessment_and_plan']


# %%
def add_section_divisions(row):
    text = row["note"]
    text_with_endlines = text.replace( '__lf1__', '\n' )
    detected_divisions = section_tagger.divide_note_by_metasections(text_with_endlines)
    for detected_division in detected_divisions:
        label, _, _, start, _, end = detected_division
        row[ '%s_%s' % ("reference", label)] = text_with_endlines[start:end].replace('\n', ' ')

    return row


# %%
# merge_df_with_sections = merge_df.apply(lambda x:add_section_divisions(x),axis=1)
# for section in SECTION_DIVISIONS:
#     merge_df_with_sections[f"dialogue_{section}"] = None

# %%
def get_preds(model,threshold,tokenized_input):
    input_ids = tokenized_input["input_ids"].to(device)
    token_type_ids = tokenized_input["token_type_ids"].to(device)
    attention_mask = tokenized_input["attention_mask"].to(device)
    
    model = model.to(device)
    preds_list = []
    preds = model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask).logits
    preds = preds.detach().cpu().squeeze()
    threshold_dict = threshold
    for threshold_idx,(section,score) in enumerate(threshold_dict.items()):
        if preds[threshold_idx].item() > float(score):
            preds_list.append(idx2label[str(threshold_idx)])
    return preds_list


# %%
def create_section_wise_dialogue_and_reference(df):
    df = df.apply(lambda x:add_section_divisions(x),axis=1)
    for section in SECTION_DIVISIONS:
        df[f"dialogue_{section}"] = None
    
    encounter_section_utterance_dict = dict()
    for row_idx,row in tqdm(df[["encounter_id","dialogue"]].iterrows()):
        encounter_id = row["encounter_id"]
        dialogue = row["dialogue"]
        encounter_section_utterance_dict[encounter_id] = dict()    
        for idx,section in idx2label.items():
            # Capture utterance_idx for every section
            encounter_section_utterance_dict[encounter_id][section] = []
        for utterance_idx,utterance in enumerate(dialogue.split("\n")):
            tokenized_sentence = \
            tokenizer.encode_plus(utterance, \
                          add_special_tokens=True, \
                          padding="max_length", \
                          truncation=True, \
                          max_length=code_config.MULTI_LABEL_MAX_LENGTH, \
                          verbose=False, \
                          return_tensors="pt", \
                          return_attention_mask=True)

            preds_list = []
            for split in [0,1,2,3,4]:
                temp_preds_list = get_preds(model_dict[split],threshold_dict[split],tokenized_sentence)
                preds_list.extend(temp_preds_list)
            preds_list = list(set(preds_list))
            for pred_section in preds_list:
                encounter_section_utterance_dict[encounter_id][pred_section].append(utterance_idx)
                
    encounter_first_level_utterance_dict = dict()
    for encounter_id,encounter_data in encounter_section_utterance_dict.items():
        encounter_first_level_utterance_dict[encounter_id] = dict()
        for section, utterance_list in encounter_data.items():
            for first_level,second_level_list in first_level_dict.items():
                for second_level in second_level_list:
                    if section == second_level:
                        if first_level not in encounter_first_level_utterance_dict[encounter_id]:
                            encounter_first_level_utterance_dict[encounter_id][first_level] = utterance_list
                        else:
                            encounter_first_level_utterance_dict[encounter_id][first_level].extend(utterance_list)

    for encounter_id,encounter_data in encounter_first_level_utterance_dict.items():
        for first_level, first_level_list in encounter_data.items():
            encounter_first_level_utterance_dict[encounter_id][first_level] = sorted(list(set(first_level_list)))
            
    encounter_first_level_conversation_dict = dict()
    for encounter_id, first_level_data in encounter_first_level_utterance_dict.items():
        encounter_first_level_conversation_dict[encounter_id] = dict()
        dialogue = df.loc[df["encounter_id"] == encounter_id, "dialogue"].item()
        utterance_list = dialogue.split("\n")
        for first_level_section, first_level_utterances in first_level_data.items():
            first_level_section_str = ""
            for utterance_idx in first_level_utterances:
                    first_level_section_str = \
                    first_level_section_str + " " + utterance_list[utterance_idx]
            encounter_first_level_conversation_dict[encounter_id][first_level_section] = \
            first_level_section_str
            
    for encounter_id, first_level_conversations in encounter_first_level_conversation_dict.items():
        for first_level_section, first_level_conversation in first_level_conversations.items():
            df.loc[df["encounter_id"] == encounter_id, \
                   f"dialogue_{first_level_section}"] = first_level_conversation
            
    return df

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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# %%


def postprocess_text(preds,labels):
    seed_everything(code_config.TASKB_SUMMARY_SEED)
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    
    return preds, labels


# %%
tokenizer_model_mapping = \
{
    "subjective":[
        ("MingZhong/DialogLED-large-5120", \
         "suryakiran786/5-KFold-dialogled-large-with-section-information-subjective"), \
        ("MingZhong/DialogLED-base-16384", \
         "suryakiran786/5-KFold-dialogled-base-with-section-information-subjective")
    ],
    "assessment_and_plan":[
        ("MingZhong/DialogLED-large-5120", \
         "suryakiran786/5-KFold-dialogled-large-with-section-information-assessment_and_plan"), \
        ("MingZhong/DialogLED-base-16384", \
         "suryakiran786/5-KFold-dialogled-base-with-section-information-assessment_and_plan")
    ],
    "objective_results":[
        ("MingZhong/DialogLED-large-5120", \
         "suryakiran786/5-KFold-dialogled-large-with-section-information-objective_results"), \
        ("MingZhong/DialogLED-base-16384", \
         "suryakiran786/5-KFold-dialogled-base-with-section-information-objective_results")
    ],
    "objective_exam":[
        ("MingZhong/DialogLED-large-5120", \
         "suryakiran786/5-KFold-dialogled-large-with-section-information-objective_exam"), \
        ("MingZhong/DialogLED-base-16384", \
         "suryakiran786/5-KFold-dialogled-base-with-section-information-objective_exam")
    ],
}

# %%
min_length_by_section = dict()
max_length_by_section = dict()
min_length_by_section["subjective"] = code_config.TASKB_SUBJECTIVE_MIN_TARGET_LENGTH
max_length_by_section["subjective"] = code_config.TASKB_SUBJECTIVE_MAX_TARGET_LENGTH
min_length_by_section["assessment_and_plan"] = code_config.TASKB_ASSESSMENT_AND_PLAN_MIN_TARGET_LENGTH
max_length_by_section["assessment_and_plan"] = code_config.TASKB_ASSESSMENT_AND_PLAN_MAX_TARGET_LENGTH
min_length_by_section["objective_results"] = code_config.TASKB_OBJECTIVE_RESULT_MIN_TARGET_LENGTH
max_length_by_section["objective_results"] = code_config.TASKB_OBJECTIVE_RESULT_MAX_TARGET_LENGTH
min_length_by_section["objective_exam"] = code_config.TASKB_OBJECTIVE_EXAM_MIN_TARGET_LENGTH
max_length_by_section["objective_exam"] = code_config.TASKB_OBJECTIVE_EXAM_MAX_TARGET_LENGTH


# %%
def optimize_summarization(df, section, \
                           tokenizer_name, model_name, \
                           dialog_column, summary_column, \
                           min_target_length, max_target_length):
    
    df["predicted_section_text_postprocessed"] = None
    df["reference_section_text_postprocessed"] = None
    
    @wandbc.track_in_wandb()
    def objective(trial):
        
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
            )
        }
        
        early_stopping = trial.suggest_categorical("early_stopping",[True])
        num_beams = trial.suggest_int("num_beams",5,15)
        no_repeat_ngram_size = trial.suggest_int("no_repeat_ngram_size",5,15)
        length_penalty = trial.suggest_float("length_penalty",-2,2,step=0.1)
        
        
        generate_kwargs = {
            "early_stopping": early_stopping, \
            "min_length": min_target_length, \
            "max_length": max_target_length, \
            "num_beams": num_beams, \
            "length_penalty": length_penalty, \
            "no_repeat_ngram_size": no_repeat_ngram_size
#             "num_beam_groups": num_beam_groups
        }
            
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
        model = model.to(device)
        model.eval()
        
        df[dialog_column] = \
        section.lower() + \
        f" {str(tokenizer.sep_token)} " + \
        df[dialog_column]
        
        for idx in tqdm(df.index):
            sentence = df.loc[idx,dialog_column]
            target = df.loc[idx,summary_column]
    
            model_inputs = \
            tokenizer(sentence, \
                      padding=code_config.TASKB_SUMMARY_PADDING, \
                      truncation=True, \
                      max_length=code_config.TASKB_SUMMARY_MAX_SOURCE_LENGTH, \
                      return_tensors="pt")
        
            labels = tokenizer(text_target=target, \
                               padding=code_config.TASKB_SUMMARY_PADDING, \
                               truncation=True, \
                               max_length=max_target_length, \
                               return_tensors="pt")
        
            input_ids = model_inputs["input_ids"].to(device)
            attention_mask = model_inputs["attention_mask"].to(device)

            with torch.no_grad():
                generated_tokens = \
                model.generate(inputs=input_ids, \
                               attention_mask=attention_mask, \
                               **generate_kwargs)

            if isinstance(generated_tokens,tuple):
                generated_tokens = generated_tokens[0]
                
            generated_tokens_decoded = tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
            labels_w_padding_tokens = labels["input_ids"]
            labels_decoded = \
            tokenizer.batch_decode(labels_w_padding_tokens,skip_special_tokens=True)

            generated_tokens_decoded,labels_decoded = \
            postprocess_text(generated_tokens_decoded,labels_decoded)

            df.loc[idx,"predicted_section_text_postprocessed"] = \
            generated_tokens_decoded[0]

            df.loc[idx,"reference_section_text_postprocessed"] = \
            labels_decoded[0]
            
        references = df['reference_section_text_postprocessed'].tolist()
        predictions = df['predicted_section_text_postprocessed'].tolist()
        df['dataset'] = 0
        num_test = len(df)
        
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
        bert_score = outputs["all"]["bertscore_f1"]
        model = model.cpu()
        del model
        gc.collect()
        time.sleep(2)
        
        return rouge1,rouge2,bert_score
    
    return objective

# %%
kfold = KFold(n_splits=code_config.TASKB_SUMMARY_N_SPLITS,shuffle=True,random_state=code_config.SEED)
for split,(train_idx,valid_idx) in enumerate(kfold.split(merge_df)):
    valid_df = merge_df.iloc[valid_idx,:]
    valid_df = valid_df.apply(lambda x:add_section_divisions(x),axis=1)
    for section in SECTION_DIVISIONS:
        valid_df[f"dialogue_{section}"] = None
    valid_df = create_section_wise_dialogue_and_reference(valid_df)
    
    for section, tokenizer_model_list in tokenizer_model_mapping.items():
        dialogue_column = f"dialogue_{section}"
        reference_column = f"reference_{section}"
        valid_df_new = valid_df[[dialogue_column,reference_column]].dropna()
        
        for tokenizer_model in tokenizer_model_list:
            tokenizer_name = tokenizer_model[0]
            model_name = f"{tokenizer_model[1]}-{split}"
            if "DialogLED-base" not in tokenizer_name:
                continue
            objective_fn = \
            optimize_summarization(valid_df_new, section, \
                                   tokenizer_name, \
                                   model_name, \
                                   dialogue_column, \
                                   reference_column, \
                                   min_length_by_section[section], \
                                   max_length_by_section[section])    
            
            study_name = \
            "beam-search" + \
            "-" + \
            tokenizer_name.split("/")[-1] + \
            "-" + \
            section + \
            "-" + \
            str(split)
            
            study = optuna.create_study(study_name=study_name, \
                                        directions=["maximize","maximize","maximize"])
            study.optimize(objective_fn, n_trials=40,callbacks=[wandbc])
