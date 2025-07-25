import itertools

import pandas as pd

import numpy as np

import datasets

import torch

import datasets

def get_INPUT_CFs(OneEntryArgs):
    Input_Part = OneEntryArgs['Input_Part']
    CF_list = Input_Part['CF_list']
    ############################ # INPUT_CFs
    assert type(CF_list) == list, f'InputCFs must be a list, but got {type(CF_list)}'
    # INPUT_CFs = sorted(InputCFs_Args)
    INPUT_CFs = CF_list

    InferenceMode = Input_Part['InferenceMode'] 
    BeforePeriods = Input_Part['BeforePeriods']
    # TargetField = Input_Part['TargetField']
    if InferenceMode == True:
        INPUT_CFs = [i for i in INPUT_CFs if any([j in i for j in BeforePeriods])]

    ############################
    return INPUT_CFs


def tfm_fn_AIInputData(examples, OneEntryArgs, CF_to_CFvocab):
    # 1. grab your input CF names and the target‐range bounds
    INPUT_CFs    = get_INPUT_CFs(OneEntryArgs)                           # e.g. ['CGMValueBf24h', …]
    low, high    = OneEntryArgs['Input_Part']['TargetRange']             # e.g. [40, 400]

    # 2. pull out the raw "--tid" lists for each CF
    #    examples[f"{cf}--tid"] is assumed to be a list of lists (len = batch size)
    tid_lists = [examples[f"{cf}--tid"] for cf in INPUT_CFs]

    # 3. for each example in the batch, clamp each sequence to [low,high] and flatten
    #    we do this all in Python lists + numpy.clip, which is far faster than DataFrame/apply
    flat_seqs = []
    for per_cf_seqs in zip(*tid_lists):
        # per_cf_seqs is a tuple like (seq_cf1, seq_cf2, …) for one example
        clamped = []
        for seq in per_cf_seqs:
            # numpy.clip can work on any sequence type
            arr = np.clip(seq, low, high)
            clamped.extend(arr.tolist())
        flat_seqs.append(clamped)

    # 4. stack into one LongTensor [batch_size, total_seq_length]
    input_ids = torch.tensor(flat_seqs, dtype=torch.long)

    # length_each_cf = [len(i) for i in tid_lists[0]]
    now_list = examples['ObsDT']
    HM_seq_list = []
    HM_args = OneEntryArgs['Input_Part'].get('HM', None)


    CFName = 'HM5MinStep'

    # columns_tid = [i for i in examples if '--tid' in i and CFName in i]

    tkn2idx = CF_to_CFvocab[CFName]['tkn2idx']
    if HM_args is not None:
        HM_start = HM_args['start']
        HM_unit = HM_args['unit']
        HM_interval = HM_args['interval']
        if HM_interval == '5m':
            interval_delta = pd.Timedelta(minutes=5)
        else:
            raise ValueError(f"Not implemented interval: {HM_interval}")

        for now in now_list:
            # HM_now = f'{now.hour}:{now.minute}'
            HM_start_dt = now + pd.Timedelta(value=HM_start, unit=HM_unit)

            length = len(input_ids[0])


            # HM_end = now + pd.Timedelta(hours=HM_end)
            # HM_now = f'{now.hour}:{now.minute}'

            HM_seq = [HM_start_dt + i * interval_delta for i in range(length)]
            HM_seq = [tkn2idx[f'{i.hour:02d}:{i.minute:02d}'] for i in HM_seq]
            HM_seq_list.append(HM_seq)

    return {
        'input_ids': input_ids,
        'hm_ids': torch.tensor(HM_seq_list, dtype=torch.long),
        # you could also add labels here, e.g.
        # 'labels': input_ids.clone()
    }


def entry_fn_AIInputData(Data, 
                         CF_to_CFvocab, 
                         OneEntryArgs,
                         tfm_fn_AIInputData = None):

    # Input feaures. 
    # INPUT_CFs = get_INPUT_CFs(OneEntryArgs)
    # print(INPUT_CFs)
    transform_fn = lambda examples: tfm_fn_AIInputData(examples, OneEntryArgs, CF_to_CFvocab)
    # ds_case 
    ds_case = Data['ds_case']
    if type(ds_case) == pd.DataFrame:
        ds_case = datasets.Dataset.from_pandas(ds_case) 
    ds_case.set_transform(transform_fn)
    ds_tfm = ds_case
    Data['ds_tfm'] = ds_tfm
    return Data


MetaDict = {
	"get_INPUT_CFs": get_INPUT_CFs,
	"tfm_fn_AIInputData": tfm_fn_AIInputData,
	"entry_fn_AIInputData": entry_fn_AIInputData
}