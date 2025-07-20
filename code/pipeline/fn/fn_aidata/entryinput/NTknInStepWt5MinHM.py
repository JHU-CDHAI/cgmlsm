import itertools

import pandas as pd

import numpy as np

import datasets

import torch

import datasets

def get_INPUT_CFs(OneEntryArgs):
    Input_Part = OneEntryArgs['Input_Part']
    TargetCFs = Input_Part['TargetCFs']

    EventCFs = Input_Part['EventCFs']
    EventCFs_list = list(EventCFs.values())
    EventCFs_list = [i for j in EventCFs_list for i in j]

    CF_list = TargetCFs + EventCFs_list
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
    # ========== 1. Prepare continuous CFs (e.g., CGM values) ==========
    TargetCFs = OneEntryArgs['Input_Part']['TargetCFs']
    low, high = OneEntryArgs['Input_Part']['TargetRange']

    # Collect and clip each input CF sequence to the target range
    tid_lists = [examples[f"{cf}--tid"] for cf in TargetCFs]
    flat_seqs = []
    for per_example in zip(*tid_lists):  # for each example in the batch
        clamped = []
        for seq in per_example:  # for each CF sequence
            clamped.extend(np.clip(seq, low, high).tolist())
        flat_seqs.append(clamped)

    # Convert to LongTensor: shape [batch_size, total_seq_length]
    input_ids = torch.tensor(flat_seqs, dtype=torch.long)
    examples_tfm = {'input_ids': input_ids}

    # ========== 2. Generate HM (Hour-Minute) Token IDs ==========
    now_list = examples['ObsDT']
    HM_args = OneEntryArgs['Input_Part'].get('HM', None)
    CFName = 'HM5MinStep'

    if HM_args is not None:
        tkn2idx = CF_to_CFvocab[CFName]['tkn2idx']
        HM_start = HM_args['start']
        HM_unit = HM_args['unit']
        interval_delta = pd.Timedelta(minutes=5)

        seq_len = input_ids.size(1)
        hm_ids_list = []

        for now in now_list:
            start_time = now + pd.Timedelta(value=HM_start, unit=HM_unit)
            times = [start_time + i * interval_delta for i in range(seq_len)]
            ids = [tkn2idx[f"{t.hour:02d}:{t.minute:02d}"] for t in times]
            hm_ids_list.append(ids)

        examples_tfm['hm_ids'] = torch.tensor(hm_ids_list, dtype=torch.long)

        # Time step relative positions (e.g., -288 to +24)
        start_timestep = HM_args['start_timestep']
        timestep_ids = torch.tensor(
            np.tile(start_timestep + np.arange(seq_len), (len(input_ids), 1)),
            dtype=torch.long
        )
        examples_tfm['timestep_ids'] = timestep_ids

    # ========== 3. Process Event-based Inputs (e.g., Diet Events) ==========
    for field, field_CFs in OneEntryArgs['Input_Part']['EventCFs'].items():
        tid_lists = [examples[f"{cf}--tid"] for cf in field_CFs]
        timestep_lists = [examples[f"{cf}--timestep"] for cf in field_CFs]

        per_sample_events = []
        all_event_token_lens = []

        # First pass: collect and compute max token length across batch
        for example_event_seqs in zip(*tid_lists):
            events = [torch.tensor(seq, dtype=torch.long) for cf_seqs in example_event_seqs for seq in cf_seqs]
            all_event_token_lens.extend([len(e) for e in events])
            per_sample_events.append(events)

        max_token_len = max(all_event_token_lens)
        max_events = max(len(evs) for evs in per_sample_events)
        batch_size = len(per_sample_events)

        # Allocate and fill padded tensor [B, max_events, max_token_len]
        field_tensor = torch.zeros((batch_size, max_events, max_token_len), dtype=torch.long)
        for i, events in enumerate(per_sample_events):
            for j, event in enumerate(events):
                field_tensor[i, j, :len(event)] = event
        examples_tfm[f"{field}_ids"] = field_tensor

        # Prepare and pad timestep_ids for each event
        merged_timesteps = []
        for example_timesteps in zip(*timestep_lists):  # zip across CFs
            merged = []
            for ts in example_timesteps:
                merged.extend(ts)
            merged_timesteps.append(torch.tensor(merged, dtype=torch.long))

        padded_timesteps = torch.nn.utils.rnn.pad_sequence(
            merged_timesteps, batch_first=True, padding_value=-9999
        )
        examples_tfm[f"{field}_timestep_ids"] = padded_timesteps

    return examples_tfm


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