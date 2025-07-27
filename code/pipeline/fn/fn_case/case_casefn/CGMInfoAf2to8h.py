import pandas as pd

import numpy as np

CaseFnName = "CGMInfoAf2to8h"

Ckpd_to_CkpdObsConfig = {'Af2to8H': {'DistStartToPredDT': 121,
             'DistEndToPredDT': 481,
             'TimeUnit': 'min',
             'StartIdx5Min': 25,
             'EndIdx5Min': 96}}

RO_to_ROName = {'RO': 'hP.rCGM5Min.cAf2to8H'}

ROName_to_RONameInfo = {'hP.rCGM5Min.cAf2to8H': {'HumanName': 'P', 'RecordName': 'CGM5Min', 'CkpdName': 'Af2to8H'}}

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
    ROdf = ROName_to_ROData[ROName]
    d = {}
    if ROdf is not None:

        # TIR_series = ROdf['BGValue'].apply(lambda x: int(x >= 70 and x<= 180))
        # TIR = TIR_series.values.mean()
        # VeryLow_series = ROdf['BGValue'].apply(lambda x: int(x < 54))
        # VeryLow = VeryLow_series.values.mean()
        # Low_series = ROdf['BGValue'].apply(lambda x: int(x >= 54 and x < 70))
        # Low = Low_series.values.mean()
        # High_series = ROdf['BGValue'].apply(lambda x: int(x > 180 and x <= 250))
        # High = High_series.values.mean()
        # VeryHigh_series = ROdf['BGValue'].apply(lambda x: int(x > 250))
        # VeryHigh = VeryHigh_series.values.mean()
        # TAR_series = ROdf['BGValue'].apply(lambda x: int(x > 180))
        # TAR = TAR_series.values.mean()
        # TBR_series = ROdf['BGValue'].apply(lambda x: int(x < 70))
        # TBR = TBR_series.values.mean()

        ModeValue = max(ROdf['BGValue'].mode()[0], 0.000001)
        ModePercent = max((ROdf['BGValue'] == ModeValue).mean(), 0.000001)


        Geq400Percent = (ROdf['BGValue'] >= 400).mean()
        Leq20Percent  = (ROdf['BGValue'] <= 20).mean()


        ZeroPercent = (ROdf['BGValue'] == 0).mean()
        # create a dictionary to store the weight for tkn list, d can be different than vocab, but in this case, they are the same. 
        recnum = len(ROdf)
        d = {
            #  'VeryLow': VeryLow, 'Low': Low, 'TIR': TIR, 'High': High, 'VeryHigh': VeryHigh, 
            #  'TAR': TAR, 
            #  'TBR': TBR, 
            #  'ModeValue': ModeValue, 
             'ModePercent': ModePercent, 
             'ZeroPercent': ZeroPercent,
             'Geq400Percent': Geq400Percent,
             'Leq20Percent': Leq20Percent,
             'RecNum': recnum}      



        d_new = {k: round(v, 6) for k, v in d.items()}
        # tkn = list(d.keys())
        # tid = [COVocab['tkn2tid'][i] for i in tkn]
        # wgt = list(d.values())
        # wgt = [round(i,4) for i in wgt]

        # d_new['-tid'] = tid
        # d_new['-wgt'] = wgt

        # 
        # COData = {'tid': tid, 'wgt': wgt}

    else:
        # tkn2tid = COvocab['tid']['tkn2tid']
        # # d['NoObs'] = 1
        # tkn = ['NoObs']
        # tid = [tkn2tid[i] for i in tkn]
        # wgt = [1]
        # CO = {'tid': tid, 'wgt': float(wgt)}
        # COData = {'tid': tid, 'wgt': [float(i) for i in wgt]}

        d = {
            #  'VeryLow': 0, 'Low': 0, 'TIR': 0, 'High': 0, 'VeryHigh': 0, 
            #  'TAR': 0, 
            #  'TBR': 0, 
            #  'ModeValue': 0, 
             'ModePercent': 0, 
             'ZeroPercent': 0,
             'Geq400Percent': 0,
             'Leq20Percent': 0,
             'RecNum': 0}  

        d_new = { k: round(v, 6) for k, v in d.items()}
        # tkn = list(d.keys())
        # tid = [COVocab['tkn2tid'][i] for i in tkn]
        # wgt = list(d.values())
        # wgt = [round(i,4) for i in wgt]
        # d_new['-tid'] = tid
        # d_new['-wgt'] = wgt 

    return d_new


MetaDict = {
	"CaseFnName": CaseFnName,
	"Ckpd_to_CkpdObsConfig": Ckpd_to_CkpdObsConfig,
	"RO_to_ROName": RO_to_ROName,
	"ROName_to_RONameInfo": ROName_to_RONameInfo,
	"HumanRecordRecfeat_Args": HumanRecordRecfeat_Args,
	"COVocab": COVocab,
	"fn_CaseFn": fn_CaseFn
}