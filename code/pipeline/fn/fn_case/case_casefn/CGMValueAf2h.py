import pandas as pd

import numpy as np

CaseFnName = "CGMValueAf2h"

Ckpd_to_CkpdObsConfig = {'Af2H': {'DistStartToPredDT': 1,
          'DistEndToPredDT': 121,
          'TimeUnit': 'min',
          'StartIdx5Min': 1,
          'EndIdx5Min': 24}}

RO_to_ROName = {'RO': 'hP.rCGM5Min.cAf2H'}

ROName_to_RONameInfo = {'hP.rCGM5Min.cAf2H': {'HumanName': 'P', 'RecordName': 'CGM5Min', 'CkpdName': 'Af2H'}}

HumanRecordRecfeat_Args = {'P': {'P': [], 'CGM5Min': []}}

COVocab = {'idx2tkn': [], 'tkn2tid': {}}

def fn_CaseFn(case_example,     # <--- case to process
               ROName_list,      # <--- from COName
               ROName_to_ROData, # <--- in scope of case_example
               ROName_to_ROInfo, # <--- in scope of CaseFleshingTask
               COVocab,          # <--- in scope of CaseFleshingTask, from ROName_to_ROInfo
               caseset,          # <--- in scope of CaseFleshingTask,
               ):

    assert len(ROName_list) == 1
    ROName = ROName_list[0]

    #############################################

    ROInfo = ROName_to_ROInfo[ROName]
    CkpdInfo = ROInfo['CkpdInfo']  # Ckpd
    StartIdx5Min = CkpdInfo['StartIdx5Min'] 
    EndIdx5Min = CkpdInfo['EndIdx5Min']

    ROData = ROName_to_ROData[ROName] # dataframe: RecObsName: Rx-bf24.. RecObsDS: the df: record collection

    if ROData is not None:
        record = ROInfo['record']
        ObsDTName = caseset.ObsDTName
        ObsDTValue = case_example[ObsDTName]

        # 1. get a subset of a Record Type: e.g., CGM5Min_Bf24H, with TknIdx.
        RecDT = record.OneRecord_Args['RecDT'] 

        # 2. get the 5MinLoc, generate some time_location features (optional)
        df = ROData.reset_index(drop = True)
        df['timestep'] = ((df[RecDT] - ObsDTValue).dt.total_seconds() / (60 * 5)).astype(int)

        # 3. filling with the empty time_location to get the full range of cgm. 
        new_index = range(StartIdx5Min, EndIdx5Min + 1)  # Include 24
        desired_range_df = pd.DataFrame({'timestep': new_index})
        df = pd.merge(df, desired_range_df, on='timestep', how='right')
        df = df.fillna(0)


        # 4. select columns
        df['tid'] = df['BGValue'].astype(int)# .to_list()
        df = df[['timestep', 'tid']]

        # 5. Explode list columns if needed
        EXPLODE_COLS = ['tid']  # 'TknInFld' included as it's okay even if not used in the model
        df = df.apply(lambda col: col.explode() if col.name in EXPLODE_COLS else col).reset_index(drop=True)
        output = df.to_dict(orient='list')
        COData = {'-tid': output['tid']}
    else:
        new_index = range(StartIdx5Min, EndIdx5Min + 1)  # Include 24
        # print('RO_ds is None')
        UNK_ID = 0
        COData = {'-tid': [UNK_ID] * len(new_index)}
    return COData


MetaDict = {
	"CaseFnName": CaseFnName,
	"Ckpd_to_CkpdObsConfig": Ckpd_to_CkpdObsConfig,
	"RO_to_ROName": RO_to_ROName,
	"ROName_to_RONameInfo": ROName_to_RONameInfo,
	"HumanRecordRecfeat_Args": HumanRecordRecfeat_Args,
	"COVocab": COVocab,
	"fn_CaseFn": fn_CaseFn
}