import pandas as pd

import numpy as np

CaseFnName = "DietNameTextAf2to8h"

Ckpd_to_CkpdObsConfig = {'Af2to8H': {'DistStartToPredDT': 121,
             'DistEndToPredDT': 481,
             'TimeUnit': 'min',
             'StartIdx5Min': 25,
             'EndIdx5Min': 96}}

RO_to_ROName = {'Diet': 'hP.rDiet5Min.cAf2to8H'}

ROName_to_RONameInfo = {'hP.rDiet5Min.cAf2to8H': {'HumanName': 'P', 'RecordName': 'Diet5Min', 'CkpdName': 'Af2to8H'}}

HumanRecordRecfeat_Args = {'P': {'P': [], 'Diet5Min': []}}

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
    df = ROName_to_ROData[ROName]

    ObsDT = case_example['ObsDT']
    tkn2idx = COVocab['tkn2tid']

    if df is not None and len(df) > 0:
        str_all = []
        fiveminutes_all = []
        for idx, rec in df.iterrows():
            DT_s = rec['DT_s']
            # print(DT_s)
            # print(ObsDT)

            time_delta = DT_s - ObsDT
            minutes = time_delta.total_seconds() / 60
            fiveminute_index = int(minutes / 5)
            # print(f"Time difference in minutes: {minutes}")
            # print(f'The 5 minute index is: {fiveminute_index}')

            # d = RecFeat_Tokenizer_fn(rec, Attr_to_AttrConfig)
            # print(d)
            # tkn = d['tkn']
            # print(tkn)

            # tid = [tkn2idx[i] for i in tkn]
            text = rec['FoodName']
            str_all.append(text)
            fiveminutes_all.append(fiveminute_index)


        output = {
            '-str': str_all,
            '-timestep': fiveminutes_all,
        }

        # pprint(output)
        # make sure the d_total's keys are consistent.  
        #############################################
    else:
        output = {
            '-str': [],
            '-timestep': [],
        }
    return output


MetaDict = {
	"CaseFnName": CaseFnName,
	"Ckpd_to_CkpdObsConfig": Ckpd_to_CkpdObsConfig,
	"RO_to_ROName": RO_to_ROName,
	"ROName_to_RONameInfo": ROName_to_RONameInfo,
	"HumanRecordRecfeat_Args": HumanRecordRecfeat_Args,
	"COVocab": COVocab,
	"fn_CaseFn": fn_CaseFn
}