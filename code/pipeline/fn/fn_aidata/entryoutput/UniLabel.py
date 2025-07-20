import torch

import pandas as pd

import numpy as np

import datasets

def get_OUTPUT_CFs(OneEntryArgs):
    if 'Output_Part' not in OneEntryArgs:
        return []
    else:
        CF_Label = OneEntryArgs['Output_Part'].get('CF_Label', [])
        if type(CF_Label) == str:
            OUTPUT_CFs = [CF_Label]
        elif type(CF_Label) == list:
            OUTPUT_CFs = CF_Label
        return OUTPUT_CFs


def entry_fn_AITaskData(Data, 
                        CF_to_CFvocab, 
                        OneEntryArgs,
                        tfm_fn_AIInputData = None,
                        entry_fn_AIInputData = None,
                        ):
    
    Data = entry_fn_AIInputData(Data, 
                            CF_to_CFvocab, 
                            OneEntryArgs,
                            tfm_fn_AIInputData)
    
    Output_Part = OneEntryArgs['Output_Part']
    label_column = Output_Part['label_column']
    ds_case = Data['ds_case']
    
    ds_tfm = Data['ds_tfm']
    Y = np.array(ds_case[label_column])
    label_rule = Output_Part.get('label_rule', None)
    
    if label_rule is not None:
        column, operator, value = label_rule
        assert label_column == column, f'label_column {label_column} does not match column {column} in label_rule'
        if operator == '>':
            Y = (Y > value).astype(int)
        elif operator == '<':
            Y = (Y < value).astype(int)
        elif operator == '==':
            Y = (Y == value).astype(int)
        else:
            raise ValueError(f'Invalid operator {operator} in label_rule')
    ds_tfm['Y'] = Y
    Data['ds_tfm'] = ds_tfm
    return Data


MetaDict = {
	"get_OUTPUT_CFs": get_OUTPUT_CFs,
	"entry_fn_AITaskData": entry_fn_AITaskData
}