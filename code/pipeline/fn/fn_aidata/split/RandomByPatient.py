import torch

import pandas as pd

import numpy as np

import datasets

from sklearn.model_selection import train_test_split

def dataset_split_tagging_fn(df_tag, OneEntryArgs):
    Split_Part = OneEntryArgs['Split_Part']
    SplitRatio = Split_Part['SplitRatio']
    train_ratio = SplitRatio['train']
    valid_ratio = SplitRatio['valid']
    test_ratio  = SplitRatio['test']
    random_state = SplitRatio['random_state']

    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Unique PIDs
    unique_pids = df_tag['PID'].unique()

    # Split PIDs into train / valid+test, then valid / test
    train_pids, temp_pids = train_test_split(
        unique_pids, train_size=train_ratio, random_state=random_state, shuffle=True
    )
    valid_pids, test_pids = train_test_split(
        temp_pids, test_size=test_ratio / (valid_ratio + test_ratio), random_state=random_state, shuffle=True
    )

    # Create mapping
    pid_to_split = {pid: 'train' for pid in train_pids}
    pid_to_split.update({pid: 'valid' for pid in valid_pids})
    pid_to_split.update({pid: 'test' for pid in test_pids})

    # Assign new column
    df_tag = df_tag.copy()
    df_tag['split'] = df_tag['PID'].map(pid_to_split)


    if Split_Part['ObsDT_Minute']:
        df_tag['ObsDT_Minute'] = df_tag['ObsDT'].dt.minute

    return df_tag


MetaDict = {
	"dataset_split_tagging_fn": dataset_split_tagging_fn
}