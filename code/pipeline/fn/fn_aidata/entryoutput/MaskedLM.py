import torch

import pandas as pd

import numpy as np

import datasets

def get_OUTPUT_CFs(OneEntryArgs):
    if 'Output_Part' not in OneEntryArgs:
        return []
    else:
        return OneEntryArgs['Output_Part'].get('CF_list', [])


def transform_fn_output(examples, 
                        tfm_fn_AIInputData, 
                        OneEntryArgs, 
                        CF_to_CFvocab):
    # print(OneEntryArgs)
    # print(OneEntryArgs['Output_Part'])
    examples_tfm = tfm_fn_AIInputData(examples, OneEntryArgs, CF_to_CFvocab)

    # print(OneEntryArgs)
    masking_rate = OneEntryArgs['Output_Part']['MaskingRate']

    TargetField = OneEntryArgs['Input_Part']['TargetField']
    TargetField_CF = [i for i in CF_to_CFvocab if TargetField in i][0]
    CFvocab = CF_to_CFvocab[TargetField_CF]
    tkn2tid = CFvocab['tkn2tid']
    mask_token_id = tkn2tid.get('MASK', 2)

    original_input_ids = examples_tfm['input_ids'].clone()
    device = original_input_ids.device

    # Create mask where tokens are selected for masking
    mask = torch.rand(original_input_ids.shape, device=device) < masking_rate
    mask_indices = mask.nonzero(as_tuple=True)
    original_token_ids = original_input_ids[mask_indices]

    # Determine replacement strategy for masked tokens
    random_tensor = torch.rand(original_token_ids.shape, device=device)
    mask_selected = (random_tensor <= 1)        

    # Apply [MASK] replacements
    replaced_token_ids = torch.where(
        mask_selected,
        torch.tensor(mask_token_id, device=device),
        original_token_ids
    )
    # Generate masked input_ids
    masked_input_ids = original_input_ids.clone()
    masked_input_ids[mask_indices] = replaced_token_ids

    # Create labels with non-masked tokens ignored (-100)
    labels = original_input_ids.clone()
    labels[~mask] = -100

    examples_tfm['input_ids'] = masked_input_ids
    examples_tfm['labels'] = labels

    # for k, v in examples.items():
    #     if k in examples_tfm: continue 
    #     examples_tfm[k] = v

    selected_columns = OneEntryArgs['Output_Part'].get('selected_columns', None)
    if selected_columns is not None:
        examples_tfm = {k: v for k, v in examples_tfm.items() if k in selected_columns}

    for i in selected_columns:
        if i not in examples_tfm:
            # print(f'{i} not in examples_tfm')
            examples_tfm[i] = examples[i]

    return examples_tfm


def entry_fn_AITaskData(Data, 
                        CF_to_CFvocab, 
                        OneEntryArgs,
                        tfm_fn_AIInputData = None,
                        entry_fn_AIInputData = None,
                        ):

    # InputCFs = OneEntryArgs['Input_FullArgs']['INPUT_CFs_Args']['InputCFs']
    transform_fn = lambda examples: transform_fn_output(examples, tfm_fn_AIInputData, OneEntryArgs, CF_to_CFvocab)
    ds_case = Data['ds_case']

    if type(ds_case) == pd.DataFrame:
        ds_case = datasets.Dataset.from_pandas(ds_case)

    # ds_case.set_transform(transform_fn)
    # use_map = OneEntryArgs.get('use_map', False)
    Output_Part = OneEntryArgs['Output_Part']
    num_proc = Output_Part.get('num_proc', 4)
    set_transform = Output_Part.get('set_transform', True)
    if set_transform == True:
        ds_case.set_transform(transform_fn)
        ds_tfm = ds_case
    else:
        old_cols = ds_case.column_names
        if 'selected_columns' in Output_Part:
            old_cols = [i for i in old_cols if i not in Output_Part['selected_columns']]
        ds_tfm = ds_case.map(transform_fn, batched = True, num_proc = num_proc)
        ds_tfm = ds_tfm.remove_columns(old_cols)

    Data['ds_tfm'] = ds_tfm

    return Data


MetaDict = {
	"get_OUTPUT_CFs": get_OUTPUT_CFs,
	"transform_fn_output": transform_fn_output,
	"entry_fn_AITaskData": entry_fn_AITaskData
}