#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import pandas as pd


# In[2]:


import sys
import json
import argparse
from tqdm import tqdm

import evaluate
import pandas as pd
import numpy as np

from sectiontagger import SectionTagger


# In[3]:


print('Loading ROUGE, BERTScore, BLEURT from HuggingFace')
scorers = {
    'rouge': (
        evaluate.load('rouge'),
        {'use_aggregator': False},
        ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
        ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    ),
    'bert_scorer': (
        evaluate.load('bertscore'),
        {'model_type': 'microsoft/deberta-xlarge-mnli','batch_size':8},
        ['precision', 'recall', 'f1'],
        ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
    ),
    'bleurt': (
        evaluate.load('bleurt', config_name='BLEURT-20'),
        {},
        ['scores'],
        ['bleurt']
    ),
}


# In[4]:


def filter_and_aggregate(obj, indices):

    agg_obj = {}
    for k, v in obj.items():
        agg_obj[k] = float(np.mean([v[i] for i in indices]))
    return agg_obj


# In[5]:


baseline_path = Path.cwd().joinpath("baseline_model_summaries")


# In[7]:


def metric_calculation(df,model_name):
    references = preds_subset['section_text'].tolist()
    predictions = preds_subset[model_name].tolist()
    num_test = len(predictions)

    all_scores = {}
    for name, (scorer, kwargs, keys, save_keys) in tqdm(scorers.items(),desc="scorers"):
        scores = scorer.compute(references=references, predictions=predictions, **kwargs)
        for score_key, save_key in zip(keys, save_keys):
            all_scores[save_key] = scores[score_key]

    cohorts = [
                ('all', list(range(num_test))),
            ]

    outputs = {k: filter_and_aggregate(all_scores, idxs) for (k, idxs) in cohorts}

    # ###### OUTPUT TO JSON FILE ########
    fn_out = f'{model_name.split("/")[-1]}.json'
    fn_out_path = baseline_path.joinpath(fn_out)
    print(f'Saving results to {fn_out}')
    with open(fn_out_path, 'w') as fd:
        json.dump(outputs, fd, indent=4)


for f in baseline_path.glob("*.csv"):
    if "model_summaries" in f.stem:
        preds_df = pd.read_csv(f)
        preds_df.fillna("",inplace=True)
        model_prefix = "suryakiran786/5-fold-stratified-cv-"
        model_name_list = \
        [col for col in preds_df.columns if model_prefix in col]
        
        for model_name in tqdm(model_name_list):
            
            preds_subset = preds_df[["ID","section_text",model_name]]

            metric_calculation(preds_subset,model_name) 
