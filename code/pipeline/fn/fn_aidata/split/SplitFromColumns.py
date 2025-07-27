import torch

import pandas as pd

import numpy as np

import datasets

from sklearn.model_selection import train_test_split

def dataset_split_tagging_fn(df_tag, OneEntryArgs):

    Split_Part = OneEntryArgs['Split_Part']

    df_tag['cohort'] = df_tag['PID'].astype(str).str[0]
    if Split_Part.get('ObsDT_Minute', False):
        df_tag['ObsDT_Minute'] = df_tag['ObsDT'].dt.minute
    return df_tag


MetaDict = {
	"dataset_split_tagging_fn": dataset_split_tagging_fn
}