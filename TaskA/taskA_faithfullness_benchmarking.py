

# Imports
from pathlib import Path
import time
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
from sentence_transformers import SentenceTransformer, util
import openai
from fire import Fire
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
warnings.filterwarnings("ignore")
tqdm.pandas()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENAI_API_KEY"] = code_config.OPENAI_API
openai.api_key = os.getenv("OPENAI_API_KEY")

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

try:
    nltk.data.find("tokenizers/punkt")
except LookupError as e:
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def postprocess_text(preds,labels):
    seed_everything(code_config.TASKA_SUMMARY_SEED)
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    
    return preds, labels

def calculate_metrics(references,predictions,scorer,key,save_key,**kwargs):
        scores = scorer.compute(references=references, predictions=predictions, **kwargs)
        if isinstance(scores[key],list):
            if len(scores[key]) > 1:
                raise Exception("scores[key] have more than one elements")
            return scores[key][0]
        return scores[key]

def filter_and_aggregate(obj, indices):
    agg_obj = {}
    for k, v in obj.items():
        agg_obj[k] = float(np.mean([v[i] for i in indices]))
    return agg_obj

def get_logits(tokenizer_name,model_name,sentence,device):

        multi_class_config = AutoConfig.from_pretrained(model_name)
        multi_class_config.num_labels = 20

        multi_class_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, \
                                                                do_lower_case=True, \
                                                                force_download=False)

        multi_class_model = \
        AutoModelForSequenceClassification.from_pretrained(model_name, \
                                                           config=multi_class_config, \
                                                           force_download=False)

        multi_class_model = multi_class_model.to(device)
        multi_class_model.eval()

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

            preds = preds.logits.detach().cpu()
            
        return preds

# +
def get_summary(tokenizer_name,model_name,device,*sentences,**kwargs):

        taska_summary_config = AutoConfig.from_pretrained(model_name)

        taska_summary_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, \
                                                                do_lower_case=True, \
                                                                force_download=False)
        sentence_w_section_desc = \
        sentences[0] + \
        f" {str(taska_summary_tokenizer.sep_token)} " + \
        sentences[1]
        
        taska_summary_model = \
        AutoModelForSeq2SeqLM.from_pretrained(model_name, \
                                              config=taska_summary_config, \
                                              force_download=False)

        taska_summary_model = taska_summary_model.to(device)
        taska_summary_model.eval()        

        model_inputs = \
        taska_summary_tokenizer(sentence_w_section_desc, \
                                padding=code_config.TASKA_SUMMARY_PADDING, \
                                truncation=True, \
                                max_length=code_config.TASKA_SUMMARY_MAX_SOURCE_LENGTH, \
                                return_tensors="pt")

        kwargs.update({"min_length":code_config.TASKA_SUMMARY_MIN_TARGET_LENGTH})
        kwargs.update({"max_length":code_config.TASKA_SUMMARY_MAX_TARGET_LENGTH})
        print(kwargs)
        with torch.no_grad():

            input_ids = model_inputs["input_ids"].to(device)
            attention_mask = model_inputs["attention_mask"].to(device)

            generated_tokens = \
            taska_summary_model.generate(inputs=input_ids, \
                                         attention_mask=attention_mask, \
                                         **kwargs)

        if isinstance(generated_tokens,tuple):
            generated_tokens = generated_tokens[0]

        generated_tokens_decoded = \
        taska_summary_tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)

        return generated_tokens_decoded[0]
    
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


# -

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_complete(text,max_length):
    time.sleep(1)
    result = openai.Completion.create(
              model="text-davinci-003",
              prompt=text,
              temperature=0.5,
              max_tokens=max_length,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0
            )
    return result["choices"][0]["text"]


def openai_summary_generation(indices,sample_df,test_dialog,test_section_header):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    test_dialog_tokens = tokenizer(test_dialog, \
                                   truncation=True, \
                                   max_length=code_config.TASKA_SUMMARY_MAX_TARGET_LENGTH)["input_ids"]
    test_dialog = tokenizer.decode(test_dialog_tokens)

    predicted_summary_list = []
    for train_idx in indices:
        train_dialog = sample_df.loc[train_idx,"dialogue_wo_whitespaces"]
        train_section_header = sample_df.loc[train_idx,"section_header"]
        train_dialog_tokens = tokenizer(train_dialog, \
                                        truncation=True, \
                                        max_length=code_config.TASKA_SUMMARY_MAX_SOURCE_LENGTH)["input_ids"]
        train_dialog = tokenizer.decode(train_dialog_tokens)
        train_summary = sample_df.loc[train_idx,"section_text"]
        train_summary_tokens = tokenizer(train_summary, \
                                         truncation=True, \
                                         max_length=code_config.TASKA_SUMMARY_MAX_SOURCE_LENGTH)["input_ids"]
        train_summary = tokenizer.decode(train_summary_tokens)
        prompt = f"{train_dialog} belonging to {train_section_header} -> {train_summary}\n"
        prompt = prompt + f"{test_dialog} belonging to {test_section_header} ->"
        predicted_summary = \
        openai_complete(prompt,code_config.TASKA_SUMMARY_MAX_TARGET_LENGTH).strip()
        predicted_summary_list.append(predicted_summary)

    longest_summary_idx = np.argmax([len(psl) for psl in predicted_summary_list])
    combined_summary_prompt = \
    "Combined the following summaries into one cohesive summary - " + \
    "\n".join(predicted_summary_list)
    final_summary = \
    openai_complete(combined_summary_prompt,code_config.TASKA_SUMMARY_MAX_TARGET_LENGTH)
    final_summary = final_summary.strip()
    final_summary = final_summary.replace("\n","").strip()
    if len(final_summary) > 0:
        return final_summary
    return predicted_summary_list[longest_summary_idx]


with open("taska_summary_configuration_baseline.json","r") as f:
    taska_summary_configuration = json.load(f)
# A hack which will only work because we have one classifier model
taska_summary_configuration = taska_summary_configuration["Bio_ClinicalBERT"]


def faithfullness_check(summary,full_text,max_length):
    faithful_tokenizer = \
    AutoTokenizer.from_pretrained("CogComp/bart-faithful-summary-detector")
    faithful_model = \
    AutoModelForSequenceClassification.from_pretrained("CogComp/bart-faithful-summary-detector")
    faithful_model = faithful_model.to(device)
    
    test_pair = \
    faithful_tokenizer(text=summary, \
                       text_pair=full_text, \
                       return_tensors='pt', \
                       max_length=max_length, \
                       padding="max_length", \
                       truncation=True)
    test_pair = test_pair.to(device)
    faithful_score = faithful_model(**test_pair).logits.detach().cpu().numpy().squeeze(0)
    test_pair = test_pair.to("cpu")
    faithful_model = faithful_model.to("cpu")
    return  faithful_score[-1]

with open("TaskA-label2idx.json","r") as f:
    label2idx = json.load(f)

with open("TaskA-idx2label.json","r") as f:
    idx2label = json.load(f)

def benchmarking(df,bm_split):

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
    
    merge_df = df
    merge_df["dialogue_wo_whitespaces"] = merge_df["dialogue"].apply(
        lambda x: re.sub(r"[\r\n\s]+", " ", x)
    )
    merge_df["predicted_section_header"] = None
    merge_df["predicted_summary"] = None
    merge_df["model_name"] = None
        
    for idx in tqdm(merge_df.index):
        sentence = merge_df.loc[idx,"dialogue_wo_whitespaces"]
        preds_list = []
        for split in [bm_split]:
            tokenizer_name = list(TASKA_MULTI_CLASS_MODEL_MAPPING.items())[0][0]
            model_name = list(TASKA_MULTI_CLASS_MODEL_MAPPING.items())[0][1]
            model_name = f"{model_name}-{split}"
            preds = get_logits(tokenizer_name,model_name,sentence,device)
            preds_list.append(preds)
        # Ensembling BERT Models
        preds_tensor = torch.cat(preds_list,dim=0)
        preds_tensor = preds_tensor.mean(dim=0).squeeze(0)

        best_idx = np.argmax(preds_tensor).item()
        section_header = idx2label[str(best_idx)]
        merge_df.loc[idx,"predicted_section_header"] = section_header.upper()

    merge_df["predicted_section_header_desription"] = \
    merge_df["predicted_section_header"].apply(lambda x: " and ".join(section_header_mapping[x.lower()]))
    merge_df["predicted_section_header_desription"] = \
    merge_df["predicted_section_header_desription"].str.lower()

    for idx in tqdm(merge_df.index):
        dialogue_wo_whitespaces = \
        merge_df.loc[idx,"dialogue_wo_whitespaces"]
        predicted_section_header_desription = \
        merge_df.loc[idx,"predicted_section_header_desription"]
        predicted_section_header = \
        merge_df.loc[idx,"predicted_section_header"].upper()
        
        summary_dict = {}
        
        for split in [bm_split]:
            print(f"split {split}")
            for tokenizer_name,model_name in TASKA_SUMMARY_TOKENIZER_MODEL_MAPPING.items():                
                print(model_name)
                model_name_with_split = f"{model_name}-{split}"
                generate_kwargs = taska_summary_configuration.get(model_name_with_split,None)
                if generate_kwargs is None:
                    continue
                summary = \
                get_summary(tokenizer_name, \
                            model_name_with_split, \
                            device, \
                            dialogue_wo_whitespaces, \
                            predicted_section_header_desription, \
                            **generate_kwargs)
                print(model_name_with_split)
                summary_dict[model_name_with_split] = summary
                
        faithfullness_dict = {}
        for model_name,summary in summary_dict.items():
            faithfullness = faithfullness_check(summary, \
                                                dialogue_wo_whitespaces, \
                                                code_config.TASKA_SUMMARY_MAX_TARGET_LENGTH)
            faithfullness_dict[model_name] = faithfullness
            
        model_name_list,faithful_list = zip(*faithfullness_dict.items())
        
        best_faithfulness_index = np.argmax(faithful_list)
        best_model = model_name_list[best_faithfulness_index]
        best_summary = summary_dict[best_model]
        merge_df.loc[idx,"predicted_summary"] = best_summary
        merge_df.loc[idx,"model_name"] = best_model
        
#    ID,section_header,section_text,dialogue,label
    merge_df = merge_df[["ID","predicted_section_header","predicted_summary","model_name"]]
    merge_df.rename(mapper={"predicted_section_header":"section_header", \
                            "predicted_summary":"section_text"}, \
                    axis=1, \
                    inplace=True)
    merge_df.to_csv(f"TaskA_faithfulness_baseline_{split}.csv",index=False)


if __name__ == "__main__":
    train_path = Path.cwd().joinpath("mediqa-chat-data","TaskA","TaskA-TrainingSet.csv")
    validation_path = Path.cwd().joinpath("mediqa-chat-data","TaskA","TaskA-ValidationSet.csv")

    train_df = pd.read_csv(train_path,index_col="ID")
    valid_df = pd.read_csv(validation_path,index_col="ID")
    merge_df = pd.concat([train_df,valid_df],axis=0,ignore_index=True)
    merge_df.reset_index(inplace=True)
    merge_df.rename(mapper={'index':'ID'},axis=1,inplace=True)    
    merge_df["label"] = merge_df["section_header"].apply(lambda x: label2idx[x])
    
    skf = StratifiedKFold(n_splits=code_config.MULTI_CLASS_N_SPLITS, \
                      shuffle=True, \
                      random_state=code_config.SEED)
    for split,(train_idx,valid_idx) in enumerate(skf.split(merge_df,y=merge_df["label"])):
        train_df = merge_df.iloc[train_idx]
        test_df = merge_df.iloc[valid_idx]
        test_df.to_csv(f"TaskA_faithfulness_Baseline_GT_{split}.csv",index=False)
        benchmarking(test_df,split)
        print(f"{split} split done")
