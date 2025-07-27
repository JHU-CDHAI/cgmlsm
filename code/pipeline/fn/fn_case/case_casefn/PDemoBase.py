import pandas as pd

import numpy as np

CaseFnName = "PDemoBase"

Ckpd_to_CkpdObsConfig = {}

RO_to_ROName = {'RO': 'hP.rP'}

ROName_to_RONameInfo = {'hP.rP': {'HumanName': 'P', 'RecordName': 'P'}}

HumanRecordRecfeat_Args = {'P': {'P': []}}

COVocab = {'idx2tkn': ['<pad>', 'age_group-None', 'gender-None', 'disease_type-None', 'age_group-0-17',
             'age_group-18-39', 'age_group-40-64', 'age_group-65+', 'gender-1', 'gender-2',
             'gender-U', 'disease_type-T1D', 'disease_type-T2D', 'disease_type-Health',
             'disease_type-PD'],
 'tkn2tid': {'<pad>': 0,
             'age_group-None': 1,
             'gender-None': 2,
             'disease_type-None': 3,
             'age_group-0-17': 4,
             'age_group-18-39': 5,
             'age_group-40-64': 6,
             'age_group-65+': 7,
             'gender-1': 8,
             'gender-2': 9,
             'gender-U': 10,
             'disease_type-T1D': 11,
             'disease_type-T2D': 12,
             'disease_type-Health': 13,
             'disease_type-PD': 14}}

def map_age_to_group(age):
    """
    Maps an age value to an age group category.

    Args:
        age (int): The age to categorize

    Returns:
        str: Age group category ("0-17", "18-39", "40-64", or "65+")
    """
    try:
        if age < 18:
            return "0-17"
        elif age < 40:
            return "18-39"
        elif age < 65:
            return "40-64"
        elif age < 80:
            return "65+"
        else:
            return 'None'
    except:
        return 'None'


def map_disease_type(disease_type):
    if disease_type == '1.0':
        return 'T1D'
    elif disease_type == '2.0':
        return 'T2D'
    elif disease_type == '0.0':
        return 'Health'
    elif disease_type == '1.5':
        return 'PD'
    else:
        return 'None'


def map_gender(gender):
    if gender == 2:
        return '2'
    elif gender == 1:
        return '1'
    else:
        return 'U'


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
    ROData = ROName_to_ROData[ROName]
    df = ROData# .to_pandas() 
    # display(df)

    rec = df.iloc[0]
    # Define a function to map age to age group

    d = {}
    d['gender'] = map_gender(rec['Gender'])
    age = case_example['ObsDT'].year - rec['YearOfBirth']
    d['age_group'] = map_age_to_group(age)
    d['disease_type'] = map_disease_type(rec['DiseaseType'])
    # d['regimen'] = rec['MRSegmentID']

    tkn =[f'{k}-{v}' for k, v in d.items()]

    tid = [COVocab['tkn2tid'][tkn] for tkn in tkn]

    output = {}
    # output['tkn'] = tkn
    output['-tid'] = tid

    return output


MetaDict = {
	"CaseFnName": CaseFnName,
	"Ckpd_to_CkpdObsConfig": Ckpd_to_CkpdObsConfig,
	"RO_to_ROName": RO_to_ROName,
	"ROName_to_RONameInfo": ROName_to_RONameInfo,
	"HumanRecordRecfeat_Args": HumanRecordRecfeat_Args,
	"COVocab": COVocab,
	"map_age_to_group": map_age_to_group,
	"map_disease_type": map_disease_type,
	"map_gender": map_gender,
	"fn_CaseFn": fn_CaseFn
}