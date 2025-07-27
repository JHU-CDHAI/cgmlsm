import pandas as pd

import numpy as np

OneRecord_Args = {'RecordName': 'P',
 'RecID': 'PID',
 'RecIDChain': ['PID'],
 'ParentRecName': None,
 'RawHumanID': 'PatientID',
 'RecDT': None,
 'RawNameList': ['Ptt'],
 'human_group_size': 100000,
 'rec_chunk_size': 100000,
 'UseTzColName': 'UserTimeZoneOffset'}

RawName_to_RawConfig = {'Ptt': {'raw_columns': ['PatientID', 'MRSegmentID', 'MRSegmentModifiedDateTime', 'DiseaseType',
                         'Gender', 'ActivationDate', 'UserTimeZoneOffset', 'UserTimeZone',
                         'Description', 'YearOfBirth'],
         'rec_chunk_size': 100000,
         'raw_datetime_column': None,
         'raw_base_columns': ['PatientID', 'UserTimeZoneOffset', 'UserTimeZone']}}

attr_cols = ['PID', 'PatientID', 'ActivationDate', 'UserTimeZone', 'UserTimeZoneOffset', 'YearOfBirth', 'MRSegmentModifiedDateTime', 'Gender', 'MRSegmentID', 'DiseaseType']

def get_RawRecProc_for_HumanGroup(df_RawRec_for_HumanGroup, OneRecord_Args, df_Human):
    df = df_RawRec_for_HumanGroup
    # 1. filter out the records we don't need (optional) 
    # 2. create a new column for raw record id (optional)
    # 3. update datetime columns 
    column = 'ActivationDate'
    df[column] = pd.to_datetime(df[column], format='mixed')
    column = 'MRSegmentModifiedDateTime'
    df[column] = pd.to_datetime(df[column], format = 'mixed')

    column = 'DiseaseType'
    df[column] = df[column].astype(float).round(1).astype(str)


    df['UserTimeZoneOffset'] = df['UserTimeZoneOffset'].fillna(0).astype(int)
    df_RawRecProc = df
    return df_RawRecProc 


MetaDict = {
	"OneRecord_Args": OneRecord_Args,
	"RawName_to_RawConfig": RawName_to_RawConfig,
	"attr_cols": attr_cols,
	"get_RawRecProc_for_HumanGroup": get_RawRecProc_for_HumanGroup
}