import torch

import pandas as pd

import numpy as np

import datasets

def get_OUTPUT_CFs(OneEntryArgs):
    if 'Output_Part' not in OneEntryArgs:
        return []
    else:
        return OneEntryArgs['Output_Part'].get('CF_list', [])


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
        ds_tfm = ds_case.map(transform_fn, batched = True, num_proc = num_proc)

    Data['ds_tfm'] = ds_tfm

    return Data


MetaDict = {
	"get_OUTPUT_CFs": get_OUTPUT_CFs,
	"entry_fn_AITaskData": entry_fn_AITaskData
}