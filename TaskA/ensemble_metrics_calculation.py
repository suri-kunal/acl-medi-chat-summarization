#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import os
from pathlib import Path
import pandas as pd


# In[ ]:


import sys
import json
import argparse
from tqdm import tqdm

import evaluate
import pandas as pd
import numpy as np

from sectiontagger import SectionTagger
from sentence_transformers import SentenceTransformer, util

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



# In[ ]:

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


# In[ ]:


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


# In[ ]:


def filter_and_aggregate(obj, indices):

    agg_obj = {}
    for k, v in obj.items():
        agg_obj[k] = float(np.mean([v[i] for i in indices]))
    return agg_obj


# In[ ]:


baseline_path = Path.cwd().joinpath("baseline_model_summaries")


# In[ ]:


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
    return  faithful_score[-1]


# In[ ]:


def calculating_similarity(summary_list):
    
    embedding_dict = {}
    for i,summary in enumerate(summary_list):
        embeddings = sentence_model.encode(summary,convert_to_tensor=True)
        embeddings = embeddings.detach().cpu()
        embedding_dict[i] = embeddings.numpy()

    similarity_dict = {}
    for model_name_1,embeddings_1 in embedding_dict.items():
        similarity_list = []
        for model_name_2,embeddings_2 in embedding_dict.items():
            if model_name_1 != model_name_2:
                cosine_sim = util.cos_sim(embeddings_1,embeddings_2).item()
                similarity_list.append(cosine_sim)
        avg_cosine_sim = np.mean(similarity_list)
        similarity_dict[model_name_1] = avg_cosine_sim
    return similarity_dict


# In[ ]:


for f in baseline_path.glob("*best_summary_ensemble.csv"):
        split = int(f.stem.split("_")[1])
        preds_df = pd.read_csv(f)
        model_name_list = \
        ["best_summary","faithful_summary"]
        
        for ensemble_method in model_name_list:
            
            references = preds_df['section_text'].tolist()
            predictions = preds_df[ensemble_method].tolist()
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
            fn_out = f'{ensemble_method}-{split}.json'
            fn_out_path = baseline_path.joinpath(fn_out)
            print(f'Saving results to {fn_out}')
            with open(fn_out_path, 'w') as fd:
                json.dump(outputs, fd, indent=4)

