import pandas as pd

import numpy as np

CaseFnName = "DMEInfoAf2h"

Ckpd_to_CkpdObsConfig = {'Af2H': {'DistStartToPredDT': 1,
          'DistEndToPredDT': 121,
          'TimeUnit': 'min',
          'StartIdx5Min': 1,
          'EndIdx5Min': 24}}

RO_to_ROName = {'Med': 'hP.rMed5Min.cAf2H', 'Exercise': 'hP.rExercise5Min.cAf2H', 'Diet': 'hP.rDiet5Min.cAf2H'}

ROName_to_RONameInfo = {'hP.rMed5Min.cAf2H': {'HumanName': 'P', 'RecordName': 'Med5Min', 'CkpdName': 'Af2H'},
 'hP.rExercise5Min.cAf2H': {'HumanName': 'P', 'RecordName': 'Exercise5Min', 'CkpdName': 'Af2H'},
 'hP.rDiet5Min.cAf2H': {'HumanName': 'P', 'RecordName': 'Diet5Min', 'CkpdName': 'Af2H'}}

HumanRecordRecfeat_Args = {'P': {'P': [], 'Med5Min': [], 'Exercise5Min': [], 'Diet5Min': []}}

COVocab = {'idx2tkn': ['MedRecNum', 'MedFirstToNow', 'MedLastToNow', 'ExerciseRecNum', 'ExerciseFirstToNow',
             'ExerciseLastToNow', 'DietRecNum', 'DietFirstToNow', 'DietLastToNow'],
 'tkn2tid': {'MedRecNum': 0,
             'MedFirstToNow': 1,
             'MedLastToNow': 2,
             'ExerciseRecNum': 3,
             'ExerciseFirstToNow': 4,
             'ExerciseLastToNow': 5,
             'DietRecNum': 6,
             'DietFirstToNow': 7,
             'DietLastToNow': 8}}

def fn_CaseFn(case_example,     # <--- case to process
               ROName_list,      # <--- from COName
               ROName_to_ROData, # <--- in scope of case_example
               ROName_to_ROInfo, # <--- in scope of CaseFleshingTask
               COVocab,          # <--- in scope of CaseFleshingTask, from ROName_to_ROInfo
               caseset,          # <--- in scope of CaseFleshingTask,
               ):

    assert len(ROName_list) == 3
    # ROName = ROName_list[0]

    def map_ROName(ROName):
        if 'Med5Min' in ROName:
            return 'Med'
        elif 'Exercise5Min' in ROName:
            return 'Exercise'
        elif 'Diet5Min' in ROName:
            return 'Diet'
        else:
            return ROName

    #############################################
    d_total = {}

    for ROName in ROName_list:
        d = {}
        ROData = ROName_to_ROData[ROName] # dataframe: RecObsName: Rx-bf24.. RecObsDS: the df: record collection
        if ROData is not None:
            d['RecNum'] = len(ROData)
            if d['RecNum'] == 0: 
                # d['NoObs'] = 1
                d['LastToNow'] = 0.0
                d['FirstToNow'] = 0.0
            else:
                ObsDTName = caseset.ObsDTName
                ObsDTValue = case_example[ObsDTName]# .isoformat()
                # COid = (COName, ObsDTValue)
                ROInfo = ROName_to_ROInfo[ROName]
                RecDT = ROInfo['record'].RecDT
                DT_s_obs = ROData.iloc[ 0][RecDT] # the time of first records
                DT_e_obs = ROData.iloc[-1][RecDT] # pd.to_datetime(dates[idx_e-1]) # the last one smaller than idx_e
                # d['recspan'] = (DT_e_obs - DT_s_obs).total_seconds() / 60 # + 5

                # print('ObsDTValue', ObsDTValue, 'DT_e_obs', DT_e_obs)
                LastToNow   = round((ObsDTValue - DT_e_obs).total_seconds() / 60, 2)
                FirstToNow = round((ObsDTValue - DT_s_obs).total_seconds() / 60, 2)
                d['LastToNow']  = LastToNow
                d['FirstToNow'] = FirstToNow
                # d['NoObs'] = 0
        else:
            d['RecNum'] = 0
            # d['NoObs'] = 1
            d['LastToNow'] = 0.0
            d['FirstToNow'] = 0.0

            # d['recspan'] = 0
            # d['recspan_0'] = 1

        item = map_ROName(ROName)
        for k, v in d.items():
            d_total[f'{item}{k}'] = v


    # make sure the d_total's keys are consistent.  
    #############################################
    return d_total


MetaDict = {
	"CaseFnName": CaseFnName,
	"Ckpd_to_CkpdObsConfig": Ckpd_to_CkpdObsConfig,
	"RO_to_ROName": RO_to_ROName,
	"ROName_to_RONameInfo": ROName_to_RONameInfo,
	"HumanRecordRecfeat_Args": HumanRecordRecfeat_Args,
	"COVocab": COVocab,
	"fn_CaseFn": fn_CaseFn
}