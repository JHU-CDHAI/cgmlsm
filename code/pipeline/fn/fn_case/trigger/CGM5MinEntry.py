import pandas as pd

import random

import numpy as np

Trigger = "CGM5MinEntry"

Trigger_Args = {'Trigger': 'CGM5MinEntry',
 'case_id_columns': ['PID', 'ObsDT'],
 'HumanID_list': ['PID'],
 'ObsDT': 'ObsDT',
 'ROName_to_RONameArgs': {'hP.rCGM5Min': {'attribute_columns': ['PID', 'PatientID', 'DT_s', 'BGValue']}}}

def get_CaseTrigger_from_RecordBase(onecohort_record_base, Trigger_Args):

    TriggerName = Trigger_Args['Trigger']
    if TriggerName in onecohort_record_base.TriggerName_to_dfCaseTrigger:
        df_case_raw = onecohort_record_base.TriggerName_to_dfCaseTrigger[TriggerName]
    else:
        ROName_to_RONameArgs = Trigger_Args['ROName_to_RONameArgs']
        ROName_to_RODS = {}
        for ROName, ROName_Args in ROName_to_RONameArgs.items():
            RONameInfo = onecohort_record_base.parse_ROName(ROName)
            # print(RONameInfo)
            HumanName, RecordName = RONameInfo['HumanName'], RONameInfo['RecordName']
            record = onecohort_record_base.Name_to_HRF[(HumanName, RecordName)]

            attribute_columns = ROName_Args['attribute_columns']
            if hasattr(record, 'ds_RecAttr'):
                df_case = record.ds_RecAttr.select_columns(attribute_columns).to_pandas()
            else:
                df_case = record.df_RecAttr
            ROName_to_RODS[ROName] = df_case
        ROName = 'hP.rCGM5Min'
        df_case_raw = ROName_to_RODS[ROName]
        # --------------------------------------------------------------------------

    df_case = df_case_raw
    # ------------------------------ Determine the ObsDT ------------------------------
    if 'ObsDT' not in df_case.columns:
        df_case['ObsDT'] = df_case['DT_s'] # [df_case_raw['drug_name'] == 'Trulicity']
        df_case = df_case.drop(columns = 'DT_s') # .from_pandas(df_case_filter)
        df_case['ObsDT'] = pd.to_datetime(df_case['ObsDT'])
    else:
        df_case['ObsDT'] = pd.to_datetime(df_case['ObsDT'])
    # --------------------------------------------------------------------------

    # ------------------------------- Update Column Sequence ------------------------
    case_id_columns = Trigger_Args['case_id_columns']
    columns = df_case.columns 
    columns = case_id_columns + [col for col in columns if col not in case_id_columns]
    df_case = df_case[columns].reset_index(drop=True)

    df_case = df_case.groupby(case_id_columns).last().reset_index()

    random.seed(42)
    np.random.seed(42)
    # torch.manual_seed(42)
    # torch.cuda.manual_seed_all(42)
    df_case['_keep_ratio'] = np.random.rand(len(df_case))
    # -------------------- e: deal with ROName = 'hP.rCGM5Min' ------------------
    return df_case


MetaDict = {
	"Trigger": Trigger,
	"Trigger_Args": Trigger_Args,
	"get_CaseTrigger_from_RecordBase": get_CaseTrigger_from_RecordBase
}