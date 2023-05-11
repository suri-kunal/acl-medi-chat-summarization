#!/usr/bin/env python
# coding: utf-8
# %%
# Imports
from pathlib import Path
from collections import OrderedDict

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
from sectiontagger import SectionTagger
from itertools import chain
import openai
from sentence_transformers import SentenceTransformer, util
from fire import Fire
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
section_tagger = SectionTagger()
warnings.filterwarnings("ignore")
tqdm.pandas()


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dest_path = Path.cwd().joinpath("baseline_model_summaries")
dest_path.mkdir(parents=True,exist_ok=True)

# %%
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

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
def create_section_wise_dialogue(df):
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
# def calculate_metrics(references,predictions,scorer,key,save_key,**kwargs):
#         scores = scorer.compute(references=references, predictions=predictions, **kwargs)
#         if isinstance(scores[key],list):
#             if len(scores[key]) > 1:
#                 raise Exception("scores[key] have more than one elements")
#             return scores[key][0]
#         return scores[key]

# %%
# def filter_and_aggregate(obj, indices):
#     agg_obj = {}
#     for k, v in obj.items():
#         agg_obj[k] = float(np.mean([v[i] for i in indices]))
#     return agg_obj

# %%
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# %%
# def postprocess_text(preds,labels):
#     seed_everything(code_config.TASKB_SUMMARY_SEED)
#     preds = [pred.strip() for pred in preds]
#     labels = [label.strip() for label in labels]
    
#     preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
#     labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    
#     return preds, labels

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
def generate_summary(section, dialog_sentence, \
                     tokenizer_name, model_name, \
                     min_target_length, \
                     max_target_length, \
                     **generate_kwargs):
            
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
    model = model.to(device)
    model.eval()

    input_sentence = \
    section.lower() + \
    f" {str(tokenizer.sep_token)} " + \
    dialog_sentence

    model_inputs = \
    tokenizer(input_sentence, \
              padding=code_config.TASKB_SUMMARY_PADDING, \
              truncation=True, \
              max_length=code_config.TASKB_SUMMARY_MAX_SOURCE_LENGTH, \
              return_tensors="pt")

    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)

    with torch.no_grad():
        generated_tokens = \
        model.generate(inputs=input_ids, \
                       attention_mask=attention_mask, \
                       min_length=min_target_length, \
                       max_length=max_target_length, \
                       **generate_kwargs)

    if isinstance(generated_tokens,tuple):
        generated_tokens = generated_tokens[0]

    generated_tokens_decoded = tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
    
    return generated_tokens_decoded[0]


# %%
with open("taskb_summary_all_models_configuration_runtime.json","r") as f:
    taskb_summary_configuration = json.load(f)


# %%

def generating_model_summary(list_of_columns):

    def generating_model_summary_by_row(row):

        section_dict = OrderedDict()
        section_dict["Subjective"] = None
        section_dict["Exam"] = None
        section_dict["Results"] = None
        section_dict["Assessment"] = None
        summary_list = []
        for col in list_of_columns:
            if ("subjective" in col) and pd.notna(row[col]) and len(row[col]) > 0:
                section_dict["Subjective"] = row[col]
#                summary_list.append("Subjective\n\n" + row[col])
            if ("objective_exam" in col) and pd.notna(row[col]) and len(row[col]) > 0:
                section_dict["Exam"] = row[col]
#                summary_list.append("Exam\n\n" + row[col])
            if ("objective_results" in col) and pd.notna(row[col]) and len(row[col]) > 0:
                section_dict["Results"] = row[col]
#                summary_list.append("Results\n\n" + row[col])
            if ("assessment_and_plan" in col) and pd.notna(row[col]) and len(row[col]) > 0:
                section_dict["Assessment"] = row[col]
#                summary_list.append("Assessment\n\n" +row[col])
        
        for section,section_text in section_dict.items():
            if section_text is not None:
                summary_list.append(f"{section}\n\n{section_text}")
        
        return "\n\n".join(summary_list)

    return generating_model_summary_by_row

# %%
def main(df,bm_split):
        
    merge_df = df
    if merge_df["encounter_id"].nunique() != merge_df.shape[0]:
        raise Exception("Duplicate encounter_id in merge_df")
    merge_df = create_section_wise_dialogue(merge_df)
    
    for encounter_id in tqdm(merge_df["encounter_id"].unique(),desc="encounter"):
    
        for section, tokenizer_model_list in tokenizer_model_mapping.items():
            
            summary_dict = {}
            dialogue_column = f"dialogue_{section}"
            reference_column = f"reference_{section}"
            min_target_length = min_length_by_section[section]
            max_target_length = max_length_by_section[section]
            generate_kwargs = taskb_summary_configuration[section]

            test_dialogue = \
            merge_df.loc[merge_df["encounter_id"] == encounter_id,dialogue_column].item()
            
            for split in [bm_split]:
                
                for tokenizer_model in tokenizer_model_list:
                    tokenizer_name =  tokenizer_model[0]
                    model_name =  tokenizer_model[1]
                    model_name = f"{model_name}-{split}"
                    new_generate_kwargs = generate_kwargs.get(model_name,None)
                    
                    if new_generate_kwargs is None:
                        continue
                    
                    summary = \
                    generate_summary(section, test_dialogue, \
                                     tokenizer_name, model_name, \
                                     min_target_length, \
                                     max_target_length, \
                                     **new_generate_kwargs)
                    
                    summary_dict[model_name] = summary


            for model_name, model_summary in summary_dict.items():            
                merge_df.loc[merge_df["encounter_id"] == encounter_id,f"{model_name}"] = model_summary

    model_architecture_list = ["dialogled-large","dialogled-base"]

    for model_arch in model_architecture_list:
        required_columns = [col for col in merge_df.columns if model_arch in col]
        concatenating_sections = generating_model_summary(required_columns)
        merge_df[f"{model_arch}_summary"] = merge_df.apply(lambda x:concatenating_sections(x),axis=1)

    merge_df.to_csv(dest_path.joinpath(f"taskB_{split}_predicted_summary.csv"),index=False)


# %%
if __name__ == "__main__":
    
    train_file = Path.cwd().joinpath("mediqa-chat-data","TaskB","TaskB-TrainingSet.csv")
    valid_file = Path.cwd().joinpath("mediqa-chat-data","TaskB","TaskB-ValidationSet.csv")

    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)
    merge_df = pd.concat([train_df,valid_df],axis=0,ignore_index=True)
    
    kfold = KFold(n_splits=code_config.TASKB_SUMMARY_N_SPLITS,shuffle=True,random_state=code_config.SEED)
    for split,(train_idx,valid_idx) in enumerate(kfold.split(merge_df)):
        valid_df = merge_df.iloc[valid_idx,:]
        main(valid_df,split)
