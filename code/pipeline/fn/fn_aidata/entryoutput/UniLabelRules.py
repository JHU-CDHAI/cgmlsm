import torch

import pandas as pd

import numpy as np

import datasets

def get_OUTPUT_CFs(OneEntryArgs):
    if 'Output_Part' not in OneEntryArgs:
        return []
    else:
        return OneEntryArgs['Output_Part'].get('CF_list', [])


def transform_fn_output(examples, tfm_fn_AIInputData, OneEntryArgs, CF_to_CFvocab):
    # Step 1: Apply the input transformation function
    examples_tfm = tfm_fn_AIInputData(examples, OneEntryArgs, CF_to_CFvocab)

    # Step 2: Extract label rule information
    label_rule = OneEntryArgs['Output_Part']['label_rule']
    assert_list = OneEntryArgs['Output_Part']['assertion']
    # cf_list = OneEntryArgs['Output_Part']['CF_list']
    # assert len(cf_list) == 1, "Only one CF supported in this label rule setup."
    # cf_name = cf_list[0]

    # Step 3: Extract feature name from rule
    target_field = None
    for v in label_rule.values():
        if isinstance(v, tuple):
            target_field = v[0]
            break
    if target_field is None:
        raise ValueError("No valid label rule with feature found.")

    # Step 4: Extract the raw feature values from the dataset
    feature_vals = examples[target_field]
    labels = []

    # Step 5: Apply label rules
    for val in feature_vals:
        assigned = False
        for label, rule in label_rule.items():
            if isinstance(rule, tuple):
                _, op, valid_range = rule
                if op == 'in':
                    if valid_range[0] <= val <= valid_range[-1]:
                        labels.append(label)
                        assigned = True
                        break
        if not assigned:
            labels.append(-100)  # default if no rule matched

    # Step 6: Enforce assertion (optional but useful during debugging)
    # for condition in assert_list:
    #     field, op, valid_range = condition
    #     vals = examples[field]
    #     for v in vals:
    #         assert valid_range[0] <= v <= valid_range[-1], f"{v} not in {valid_range}"

    # Step 7: Format output
    examples_tfm['labels'] = torch.LongTensor(labels)
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