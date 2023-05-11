# Global Variables
SEED = 42
WANDB_API = "USE_YOUR_OWN"
HF_API = "USE_YOUR_OWN"
OPENAI_API = "USE_YOUR_OWN"

# Multi Class Classification Variables
MULTI_CLASS_WANDB_PROJECT = "mediqa-chat-multi-class-classification"
# MULTI_CLASS_WANDB_PROJECT = "mediqa-chat-mc-test"
MULTI_CLASS_EPOCHS = 30
MULTI_CLASS_BATCH_SIZE = 16
MULTI_CLASS_LEARNING_RATE = 2e-5
MULTI_CLASS_WARM_UP_STEPS = 0.1
MULTI_CLASS_ADAMW_WEIGHT_DECAY = 0.01
MULTI_CLASS_ADAMW_EPS = 1e-6
MULTI_CLASS_MAX_LENGTH = 512
MULTI_CLASS_MODEL_CHECKPOINT = "emilyalsentzer/Bio_ClinicalBERT"
MULTI_CLASS_N_SPLITS = 5
MULTI_CLASS_JOB_TYPE = "accuracy_baseline"
MULTI_CLASS_SAMPLING = False
MUTLI_CLASS_BALANCE_LOSS = False
MULTI_CLASS_SEED = 42
MULTI_CLASS_MODEL_NAME = "5-stratified-cv-bio-clinicalbert-multiclass-focal-loss-seed-42-complete-data"
MULTI_CLASS_NOTES = \
"""
Multi Class Classification with 5 fold stratified cross validation on section-header. Uses  complete data with 20 classes with Focal Loss and seed set to 42.
Accuracy for every class
"""

# +
# Multi Label Classification Variables
MULTI_LABEL_WANDB_PROJECT = "mediqa-chat-multi-label-classification"
# MULTI_LABEL_WANDB_PROJECT = "mediqa-chat-ml-test"
MULTI_LABEL_EPOCHS = 30
MULTI_LABEL_BATCH_SIZE = 8
MULTI_LABEL_LEARNING_RATE = 2e-5
MULTI_LABEL_WARM_UP_STEPS = 0.1
MULTI_LABEL_ADAMW_WEIGHT_DECAY = 0.01
MULTI_LABEL_ADAMW_EPS = 1e-6
MULTI_LABEL_MAX_LENGTH = 512
MULTI_LABEL_ATTRIBUTION_LENGTH = 200
MULTI_LABEL_MODEL_CHECKPOINT = "emilyalsentzer/Bio_ClinicalBERT"
MULTI_LABEL_N_SPLITS = 5
MULTI_LABEL_JOB_TYPE = "roc_auc_pr_baseline"
MULTI_LABEL_SAMPLING = False
MUTLI_LABEL_BALANCE_LOSS = False
MULTI_LABEL_SEED = 42
MULTI_LABEL_MODEL_NAME = "5-fold-multilabel-cv-bio-clinicalbert-multilabel-focal-loss-seed-42-complete-data"
MULTI_LABEL_NOTES = \
"""
Multi Label Classification on complete data with 20 classes with Focal Loss and seed set to 42.
MultiLabel Stratification has been used as cross validation strategy.
ROC-AUC and PR Score for every class
"""

# Summary Generation Variables
TASKA_SUMMARY_WANDB_PROJECT = "mediqa-chat-summarization"
# TASKA_SUMMARY_WANDB_PROJECT = "mediqa-chat-taska-summary-test"
TASKA_SUMMARY_EPOCHS = 30
TASKA_SUMMARY_BATCH_SIZE = 5
TASKA_SUMMARY_LEARNING_RATE = 2e-5
TASKA_SUMMARY_MAX_SOURCE_LENGTH = 512
TASKA_SUMMARY_MIN_TARGET_LENGTH = 8
TASKA_SUMMARY_MAX_TARGET_LENGTH = 400
TASKA_SUMMARY_PADDING = "max_length"
TASKA_SUMMARY_IGNORE_PAD_TOKEN_FOR_LOSS = True
TASKA_SUMMARY_MODEL_CHECKPOINT = "google/flan-t5-large"
TASKA_SUMMARY_MODEL_NAME = "5-fold-stratified-cv-flan-t5-large-with-section-description-complete-data"
TASKA_SUMMARY_PREFIX = "summarize: "
TASKA_SUMMARY_N_SPLITS = 5
TASKA_SUMMARY_NOTES = \
"""
Summarization of complete dialogues with section information.
The data has been stratified on Section Header for 5 folds
Metric for early stopping is Log Loss.
Metric for text generation evaluation is ROGUE, BERTScore, BLEURT
"""
TASKA_SUMMARY_JOB_TYPE = "rouge_bertscore_bluert_baseline"
TASKA_SUMMRAY_USE_STEMMER = True
TASKA_SUMMARY_WEIGHT_DECAY = 0.01
TASKA_SUMMARY_NUM_WARMUP_STEPS = 0.1
TASKA_SUMMARY_NUM_BEAMS = 5
TASKA_SUMMARY_DIALOGUE_W_SECTION_CODE = False
TASKA_SUMMARY_DIALOGUE_W_SECTION_CODE_DESC = True
TASKA_SUMMARY_SAMPLING = False
TASKA_SUMMARY_SEED = 42
TASKA_GRADIENT_ACCUMULATION_STEPS = 3

# # +
# Task B Summary Generation Variables
TASKB_SUMMARY_WANDB_PROJECT = "mediqa-chat-TaskB-summarization"
# TASKB_SUMMARY_WANDB_PROJECT = "mediqa-chat-TaskB-summarY-test"
TASKB_SUMMARY_EPOCHS = 30
TASKB_SUMMARY_BATCH_SIZE = 4
TASKB_SUMMARY_LEARNING_RATE = 2e-5

TASKB_SUMMARY_MAX_SOURCE_LENGTH = 3400

TASKB_SUBJECTIVE_MIN_TARGET_LENGTH = 50
TASKB_OBJECTIVE_EXAM_MIN_TARGET_LENGTH = 5
TASKB_OBJECTIVE_RESULT_MIN_TARGET_LENGTH = 5
TASKB_ASSESSMENT_AND_PLAN_MIN_TARGET_LENGTH = 50

TASKB_SUBJECTIVE_MAX_TARGET_LENGTH = 768
TASKB_OBJECTIVE_EXAM_MAX_TARGET_LENGTH = 256
TASKB_OBJECTIVE_RESULT_MAX_TARGET_LENGTH = 256
TASKB_ASSESSMENT_AND_PLAN_MAX_TARGET_LENGTH = 640

TASKB_SUMMARY_PADDING = "max_length"
TASKB_SUMMARY_IGNORE_PAD_TOKEN_FOR_LOSS = True
TASKB_SUMMARY_MODEL_CHECKPOINT = "MingZhong/DialogLED-large-5120"
TASKB_SUMMARY_MODEL_NAME = "5-KFold-dialogled-large-with-section-information"
TASKB_SUMMARY_PREFIX = ""
TASKB_SUMMARY_N_SPLITS = 5
TASKB_SUMMARY_NOTES = \
"""
Summarization of Long Dialogues with section code description. 
Early Stopping criteria is Loss
Metrics are Rouge, Bertscore, BlueRT
"""
TASKB_SUMMARY_JOB_TYPE = "rouge_bertscore_bluert_baseline"
TASKB_SUMMRAY_USE_STEMMER = True
TASKB_SUMMARY_WEIGHT_DECAY = 0.01
TASKB_SUMMARY_NUM_WARMUP_STEPS = 0.1
TASKB_SUMMARY_NUM_BEAMS = 2
TASKB_SUMMARY_DIALOGUE_W_SECTION_CODE = False
TASKB_SUMMARY_DIALOGUE_W_SECTION_CODE_DESC = True
TASKB_SUMMARY_SAMPLING = False
TASKB_SUMMARY_SEED = 42
TASKB_GRADIENT_ACCUMULATION_STEPS = 4
