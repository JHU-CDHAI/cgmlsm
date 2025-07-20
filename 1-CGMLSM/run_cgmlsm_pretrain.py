import sys
import os
import logging
import pandas as pd
import datasets
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    RobertaConfig,
    RobertaForSequenceClassification,
    PreTrainedTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from pprint import pprint
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
# Calculate AUC (Area Under the ROC Curve)
# For multi-class, we use one-vs-rest approach
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score


# ---------------------- Workspace Setup ----------------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s:%(asctime)s:(%(filename)s@%(lineno)d %(name)s)]: %(message)s')
logger = logging.getLogger(__name__)

SPACE = {
    'DATA_RAW': '_Data/0-Data_Raw',
    'DATA_RFT': '_Data/1-Data_RFT',
    'DATA_CASE': '_Data/2-Data_CASE',
    'DATA_AIDATA': '_Data/3-Data_AIDATA',
    'DATA_EXTERNAL': 'code/external',
    'DATA_HFDATA': '_Data/5-Data_HFData',
    'CODE_FN': 'code/pipeline',
    'MODEL_ROOT': '_Model',
}
assert os.path.exists(SPACE['CODE_FN']), f"{SPACE['CODE_FN']} not found"
sys.path.append(SPACE['CODE_FN'])

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---------------------- Data Preparation ----------------------
from nn.cgmlsm.configuration_cgmlsm import CgmLsmConfig
from nn.cgmlsm.modeling_cgmlsm import CgmLsmLMHeadModel

HFDataName = 'CgmLsm_WellDoc_ds0p10'
path = os.path.join(SPACE['DATA_HFDATA'], HFDataName)
split_to_dataset = datasets.load_from_disk(path)
remove_unused_columns = True # if using the processed dataset, set to True. 
print(split_to_dataset)
Name_to_Data = {i: {'ds_tfm': split_to_dataset[i]} for i in split_to_dataset}
# exit()


# ---------------------- CF Vocab Setup ----------------------
CF_to_CFvocab = {} # data_config['CF_to_CFvocab']

CFName = 'HM5MinStep'
interval_delta = pd.Timedelta(minutes=5)
idx2tkn = [pd.Timestamp('2022-01-01 00:00:00') + interval_delta * i for i in range(24 * 12)]
idx2tkn = [f'{i.hour:02d}:{i.minute:02d}' for i in idx2tkn]
tkn2idx = {tkn: idx for idx, tkn in enumerate(idx2tkn)}
CF_to_CFvocab[CFName] = {'idx2tkn': idx2tkn, 'tkn2idx': tkn2idx}

CFName = 'CGMValue'
idx2tkn = ["PAD", "UNKNOWN", "MASK"] + [f'Other_{i}' for i in range(0, 7)] + [str(i) for i in range(10, 401)]
tkn2idx = {tkn: idx for idx, tkn in enumerate(idx2tkn)}
CF_to_CFvocab[CFName] = {'idx2tkn': idx2tkn, 'tkn2idx': tkn2idx}

# ---------------------- Tokenizer Setup ----------------------
idx2tkn = CF_to_CFvocab['CGMValue']['idx2tkn']
vocab_dict = {token: idx for idx, token in enumerate(idx2tkn)}
wordlevel = WordLevel(vocab=vocab_dict, unk_token="UNKNOWN")
tokenizer_backend = Tokenizer(wordlevel)
tokenizer_backend.pre_tokenizer = Whitespace()
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_backend,
    unk_token="UNKNOWN",
    pad_token="PAD",
    mask_token="MASK"
)

# ---------------------- Model Config ----------------------
hm_idx2tkn = CF_to_CFvocab['HM5MinStep']['idx2tkn']

model_name = 'cgmlsm_pretrain_welldoc_v2024'
config = CgmLsmConfig(
    vocab_size=len(tokenizer),
)
model = CgmLsmLMHeadModel(config)
# model

print(model)

# ---------------------- Metrics Function ----------------------
# Step 1: Define the compute_metrics function for masked language modeling

def compute_metrics(pred):
    # Convert logits and labels to torch.Tensor
    logits = torch.tensor(pred.predictions)
    labels = torch.tensor(pred.label_ids)

    # Shift for next token prediction
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    # Flatten the tensors
    shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    shift_labels = shift_labels.reshape(-1)

    # Compute loss, ignoring padding tokens (-100)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    next_token_loss = loss_fct(shift_logits, shift_labels)

    # Compute accuracy
    predictions = torch.argmax(shift_logits, dim=-1)
    mask = shift_labels != -100
    correct = (predictions == shift_labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()

    # Compute perplexity
    perplexity = torch.exp(next_token_loss)

    return {
        'next_token_loss': next_token_loss.item(),
        'perplexity': perplexity.item(),
        'accuracy': accuracy.item()
    }




# ---------------------- Training Arguments ----------------------
training_args = TrainingArguments(
    output_dir=os.path.join(SPACE['MODEL_ROOT'], model_name),

    do_train=True,
    do_eval=True,

    num_train_epochs=2,  # ← First run with 1 epoch
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=2,  # effective batch size = 64*4 = 256

    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=1000,
    max_grad_norm=1.0,

    logging_steps=1,

    # Evaluation settings
    eval_strategy="steps",
    eval_steps = 200, # 1, # 200

    save_strategy="steps", # one epoch
    save_steps=0.1,
    save_total_limit=10,

    # No best model logic for now
    # load_best_model_at_end=True,
    # metric_for_best_model="perplexity",
    # greater_is_better=False,

    report_to="wandb",
    prediction_loss_only=False,
    # remove_unused_columns=False,
    remove_unused_columns=True,
    dataloader_drop_last=True,

    dataloader_num_workers=8,  # ← add this to use your CPUs
)


# ---------------------- Trainer Setup and Training ----------------------
# data_collator = DataCollatorWithPadding(tokenizer)

eval_set_size = 1042
random_seed = 42

ds_tfm_train  = Name_to_Data['In-Train']['ds_tfm']
ds_tfm_valid_t1d  = Name_to_Data['In-Valid_T1D']['ds_tfm'].shuffle(seed=random_seed).select(range(eval_set_size))
ds_tfm_valid_t2d = Name_to_Data['In-Valid_T2D']['ds_tfm'].shuffle(seed=random_seed).select(range(eval_set_size))


ds_tfm_testid_t1d  = Name_to_Data['In-Test_T1D']['ds_tfm'].shuffle(seed=random_seed).select(range(eval_set_size))
ds_tfm_testid_t2d = Name_to_Data['In-Test_T2D']['ds_tfm'].shuffle(seed=random_seed).select(range(eval_set_size))


ds_tfm_testod_t1d = Name_to_Data['Out_T1D']['ds_tfm'].shuffle(seed=random_seed).select(range(eval_set_size))
ds_tfm_testod_t2d = Name_to_Data['Out_T2D']['ds_tfm'].shuffle(seed=random_seed).select(range(eval_set_size))



eval_dict = {
    'valid_t1d': ds_tfm_valid_t1d,
    'valid_t2d': ds_tfm_valid_t2d,
    'testid_t1d': ds_tfm_testid_t1d,
    'testid_t2d': ds_tfm_testid_t2d,
    'testod_t1d': ds_tfm_testod_t1d,
    'testod_t2d': ds_tfm_testod_t2d,
}


print(ds_tfm_train)
print(eval_dict)
# exit()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_tfm_train,
    eval_dataset=eval_dict,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate(eval_dataset=ds_tfm_testid, metric_key_prefix="testid")
trainer.evaluate(eval_dataset=ds_tfm_testod, metric_key_prefix="testod")

# ---------------------- Run Script ----------------------
'''
ssh login01
tmux ls
# tmux new -s welldoc
tmux attach -t welldoc


srun --partition=ica100 --gpus=1 --mem=60GB --cpus-per-task=12 --time=1-06:00:00 --pty /bin/bash
srun --partition=a100 --gpus=1 --mem=60GB --cpus-per-task=12 --time=2-00:00:00 --pty /bin/bash
conda activate torch
cd workspace/WellDoc-SPACE



# --- a tip: set the eval_steps to be 1, and do the quick evaluation.
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT=CGMLSM-Welldoc
python 1-CGMLSM/run_cgmlsm_pretrain.py; exit


# monitor
htop -u jluo41 # change to your username
watch -n 0.1 'nvidia-smi'
'''
