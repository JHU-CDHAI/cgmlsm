import pandas as pd

import numpy as np

import logging

logger = logging.getLogger(__name__)

OneRecord_Args = {'RecordName': 'CGM5Min',
 'RecID': 'CGM5MinID',
 'RecIDChain': ['PID'],
 'RawHumanID': 'PatientID',
 'ParentRecName': 'P',
 'RecDT': 'DT_s',
 'RawNameList': ['ElogBGEntry'],
 'human_group_size': 50,
 'rec_chunk_size': 100000}

RawName_to_RawConfig = {'ElogBGEntry': {'raw_columns': ['PatientID', 'ObservationDateTime', 'BGValue', 'TimezoneOffset',
                                 'EntrySourceID', 'ExternalSourceID'],
                 'raw_base_columns': ['PatientID', 'ObservationDateTime', 'TimezoneOffset',
                                      'EntrySourceID', 'ExternalSourceID'],
                 'rec_chunk_size': 100000,
                 'raw_datetime_column': 'ObservationDateTime'}}

attr_cols = ['PID', 'PatientID', 'CGM5MinID', 'DT_s', 'BGValue']

def get_RawRecProc_for_HumanGroup(df_RawRec_for_HumanGroup, OneRecord_Args, df_Human):
    df = df_RawRec_for_HumanGroup

    # 1. filter out the records we don't need (optional) 
    logger.info('========== start processing the raw CGM data ======\n\n')

    vc = df['EntrySourceID'].value_counts()
    logger.info(vc)


    ##########
    index = (df['EntrySourceID'] == 18) # | (df['MeterType'] == 5)  # CGM
    df = df[index].reset_index(drop = True)
    ##########

    vc = df['EntrySourceID'].value_counts()
    logger.info(vc)



    ##########
    logger.info('Inpsect the TimezoneOffset')
    vc = df['TimezoneOffset'].value_counts()
    logger.info(vc)

    # df = df[df['TimezoneOffset'].abs() < 1000].reset_index(drop = True)
    OffSet_list = [-720, -660, 
               -600, -540, -480, -420, -360, -300, -240, 
               0, # we also need to keep the 0 offset
               # -180, -120, -60, 0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840
               ]
    df = df[df['TimezoneOffset'].isin(OffSet_list)].reset_index(drop = True)

    vc = df['TimezoneOffset'].value_counts()
    logger.info(vc)

    ##########

    ##########
    logger.info('Inpsect the PatientID')
    patient_counts = df['PatientID'].value_counts()
    logger.info(patient_counts)

    frequent_patients = patient_counts[patient_counts >= 288].index
    df = df[df['PatientID'].isin(frequent_patients)].reset_index(drop = True)

    patient_counts = df['PatientID'].value_counts()
    logger.info('After filtering the PatientID')
    logger.info(patient_counts)
    ##########


    # 2. create a new column for raw record id (optional)

    # 3. update datetime columns 
    DTCol_list = [
        'ObservationDateTime', 
        # 'ParentEntryID', 'ActivityTypeID',
        # 'ObservationEntryDateTime', 
        # 'EntryCreatedDateTime',
        # 'UserObservationDateTime'
        ]

    for DTCol in DTCol_list: 
        df[DTCol] = pd.to_datetime(df[DTCol], format = 'mixed')


    df['DT_tz'] = pd.Series(
        df['TimezoneOffset'].to_numpy(),
        dtype=pd.Int64Dtype()
    )


    logger.info(len(df))
    DTCol = 'DT_s'
    DTCol_source = 'ObservationDateTime'
    df[DTCol] = df[DTCol_source]
    # print(df.head())


    # df[DTCol] = pd.to_datetime(df[DTCol]).apply(lambda x: None if x <= pd.to_datetime('2019-01-01') else x)
    ## print(df[DTCol]) 
    # print(df['DT_tz'])
    # print(pd.to_timedelta(df['DT_tz'], 'm'))
    df[DTCol] = pd.to_datetime(df[DTCol]) + pd.to_timedelta(df['DT_tz'], 'm')
    logger.info('clean the data before 2019-01-01')
    logger.info(len(df))

    # print(df.head())


    df = df[df[DTCol].notna()].reset_index(drop = True)
    logger.info('drop the records with no datetime')
    logger.info(len(df))
    # assert df[DTCol].isna().sum() == 0

    DTCol_list = ['DT_s', 
                  # 'DT_r'
                  ] # 'DT_e'
    for DTCol in DTCol_list:
        # DateTimeUnit ='5Min'
        date = df[DTCol].dt.date.astype(str)
        hour = df[DTCol].dt.hour.astype(str)
        minutes = ((df[DTCol].dt.minute / 5).astype(int) * 5).astype(str)
        df[DTCol] = pd.to_datetime(date + ' ' + hour +':' + minutes + ':' + '00')

    # x3. drop duplicates
    df = df.drop_duplicates()

    # 4. select a DT as the RecDT
    RecDT = 'DT_s'
    # ----------------------------------------------------------------- #
    # x4. get the BGValue mean by RecDT (5Min)
    RawHumanID = OneRecord_Args['RawHumanID']

    logger.info('get the BGValue mean by RecDT (5Min)')
    logger.info(f'from the size of {len(df)}:')
    df = df.groupby([RawHumanID, RecDT])[['BGValue']].mean().reset_index()
    logger.info(f'to the size of {len(df)}')
    logger.info('========== finish processing the raw CGM data ======\n\n')
    # ----------------------------------------------------------------- #

    df_RawRecProc = df
    return df_RawRecProc 


MetaDict = {
	"OneRecord_Args": OneRecord_Args,
	"RawName_to_RawConfig": RawName_to_RawConfig,
	"attr_cols": attr_cols,
	"get_RawRecProc_for_HumanGroup": get_RawRecProc_for_HumanGroup
}