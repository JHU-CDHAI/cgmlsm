import sys
import os
import logging
import pandas as pd
import datasets
from pprint import pprint
KEY = '2-OhioT1DM'
WORKSPACE_PATH = os.getcwd().split(KEY)[0]
print(WORKSPACE_PATH); os.chdir(WORKSPACE_PATH)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s:%(asctime)s:(%(filename)s@%(lineno)d %(name)s)]: %(message)s')

SPACE = {
    'DATA_RAW': f'_Data/0-Data_Raw',
    'DATA_RFT': f'_Data/1-Data_RFT',
    'DATA_CASE': f'_Data/2-Data_CASE',
    'DATA_CFDATA': f'_Data/3-Data_CFDATA',
    'DATA_SPLIT': f'_Data/4-Data_Split',
    'DATA_EXTERNAL': f'code/external',
    'CODE_FN': f'code/pipeline',
    'MODEL_ROOT': f'./_Model',
}
assert os.path.exists(SPACE['CODE_FN']), f'{SPACE["CODE_FN"]} not found'
print(SPACE['CODE_FN'])
sys.path.append(SPACE['CODE_FN'])



from config.config_record.Cohort import CohortName_to_OneCohortArgs
from recfldtkn.record_base import Record_Base

###################################
Record_Proc_Config = {
    'save_data': True, 
    'load_data':True, 
    'via_method': 'ds',
    # 'shadow_df': True,
}
###################################


CohortName = 'OhioT1DM'
CohortName_list = [
    CohortName,
]


HumanRecordRecfeat_Args = {
    # Human
    'P': {
        # --------------------- patient ---------------------
        'P': [],  # patient
        'CGM5Min': [], # CGM5Min
    }
}

# HumanRecordRecfeat_Args[('P', 'CGM5Min')] = []
record_base = Record_Base(CohortName_list, 
                            HumanRecordRecfeat_Args,
                            CohortName_to_OneCohortArgs,
                            SPACE = SPACE, 
                            # Inference_Entry = Inference_Entry,
                            Record_Proc_Config = Record_Proc_Config,
                            )



one_base = record_base.CohortName_to_OneCohortRecordBase[CohortName]

human = one_base.Name_to_HRF['P']

df_Human = human.df_Human
df_Human['split'] = df_Human['PatientID'].str.split('_').str[1]
df_Human

PID_to_split = dict(zip(df_Human['PID'], df_Human['split']))
PID_to_split

from recfldtkn.aidata_base.entry import EntryAIData_Builder

CF_DataName = 'AllCGMTag-CaseBase-CGM5MinEntry-c258d484c1f72f1f'
CohortName_list = ['OhioT1DM']

CF_DataName_list = [f'{CF_DataName}/{i}' for i in CohortName_list]
entry = EntryAIData_Builder(SPACE=SPACE)
dataset = entry.merge_one_cf_dataset(CF_DataName_list)

print(dataset)

tag_columns = ['PID', 'ObsDT', 
           
           'CGMInfoBf24h-ModePercent', 
           'CGMInfoBf24h-ZeroPercent', 
           'CGMInfoBf24h-Geq400Percent',
           'CGMInfoBf24h-Leq20Percent', 
           'CGMInfoBf24h-RecNum', 

           'CGMInfoAf2h-ModePercent', 
           'CGMInfoAf2h-ZeroPercent', 
           'CGMInfoAf2h-Geq400Percent', 
           'CGMInfoAf2h-Leq20Percent', 
           'CGMInfoAf2h-RecNum', 

           'CGMInfoAf2to8h-ModePercent', 
           'CGMInfoAf2to8h-ZeroPercent', 
           'CGMInfoAf2to8h-Geq400Percent', 
           'CGMInfoAf2to8h-Leq20Percent', 
           'CGMInfoAf2to8h-RecNum', 

           
           'YearOfBirth', 
           # 'UserTimeZone', 'UserTimeZoneOffset', 
           'Gender', 'MRSegmentID', 'DiseaseType', # 'Selected'
           
           ]

df_case = dataset.select_columns(tag_columns).to_pandas()
print(df_case.shape)
df_case['Date'] = df_case['ObsDT'].dt.date
path = os.path.join(SPACE['DATA_SPLIT'], 'OhioT1DM_full.parquet')
df_case.to_parquet(path)

path = os.path.join(SPACE['DATA_SPLIT'], 'OhioT1DM_full.parquet')
df_case_all = pd.read_parquet(path)
print(df_case_all.shape)

from recfldtkn.base import apply_multiple_conditions

# df_case = df_case_hours
df_case = df_case_all

############ How to define a good case? and then define a good patient-day?
good_case_conditions = [
    ["CGMInfoBf24h-RecNum", '>=', 289],
    ["CGMInfoAf2h-RecNum", '>=', 24],
    # ["CGMInfoAf2to8h-RecNum", '>=', 12 * 6],

    ["CGMInfoBf24h-ModePercent",   '<=', 0.4],
    ["CGMInfoAf2h-ModePercent",    '<=', 0.4],
    # ["CGMInfoAf2to8h-ModePercent", '<=', 0.4],

    ["CGMInfoBf24h-Geq400Percent",   '<=', 0.2],
    ["CGMInfoAf2h-Geq400Percent",    '<=', 0.2],
    # ["CGMInfoAf2to8h-Geq400Percent", '<=', 0.2],

    ["CGMInfoBf24h-Leq20Percent",    '==', 0],
    ["CGMInfoAf2h-Leq20Percent",     '==', 0],
    # ["CGMInfoAf2to8h-Leq20Percent",  '==', 0],

]
good_case_rules = apply_multiple_conditions(df_case, good_case_conditions)

############
print(df_case.shape)
df_case_good = df_case[good_case_rules].reset_index(drop = True)
print(df_case_good.shape, len(df_case_good) / len(df_case))

def get_cutoffs(sub_df):
    sub_df = sub_df.sort_values('ObsDT').reset_index(drop=True)
    n = len(sub_df)
    idx_early_end = int(n * 0.8)
    idx_middle_end = int(n * 0.9)

    # Safely get datetime cutoffs
    first_middle_day = sub_df.loc[idx_early_end, 'ObsDT'] if idx_early_end < n else pd.NaT
    first_late_day = sub_df.loc[idx_middle_end, 'ObsDT'] if idx_middle_end < n else pd.NaT
    return first_middle_day, first_late_day


df = df_case_good
# Store results
records = []
# Apply per PID
for pid, group in df.groupby('PID'):
    mid_day, late_day = get_cutoffs(group)
    records.append({
        'PID': pid,
        'middle_first_date': mid_day,
        'late_first_date': late_day
    })

# Create result DataFrame
df_pat_dates_info = pd.DataFrame(records)
df_pat_dates_info

tag_columns = [
    'PID', #'ObsDT', 
    # 'YearOfBirth',
    # 'UserTimeZone', 'UserTimeZoneOffset', 
   #  'Gender', 'MRSegmentID', 
    'DiseaseType']
    
df_patient = df_case_all[tag_columns].drop_duplicates().reset_index(drop = True)
print(df_patient.shape)
df_patient 
df_patient['split'] = df_patient['PID'].map(PID_to_split)

df_patient_info = pd.merge(df_patient, df_pat_dates_info, on='PID')
df_patient_info

DATA_SPLIT = f'_Data/4-Data_Split'
df_patient_info.to_parquet(os.path.join(DATA_SPLIT, 'OhioT1DM_patient_split_info.parquet'))
# df_patient_info.to_csv(os.path.join(DATA_SPLIT, 'WellDoc_patient_info.csv'), index=False)


df_case_good = pd.merge(df_case_good, df_patient_info, on='PID', how='left')
df_case_good

def get_early_middle_late_label(row):

    if row['ObsDT'] >= row['middle_first_date'] and row['ObsDT'] < row['late_first_date']:
        return 'middle'
    elif row['ObsDT'] >= row['late_first_date']:
        return 'late'
    else:
        return 'early'

df_case_good['time_bin'] = df_case_good.apply(get_early_middle_late_label, axis=1)
df_case_good['time_bin'].value_counts().sort_index()

def map_split_name(row):
    if row['split'] == 'train' and row['time_bin'] in ['early', 'middle']:
        return 'train' 
    elif row['split'] == 'train' and row['time_bin'] == 'late':
        return 'valid'
    elif row['split'] == 'test':
        return 'test-id'
    else:
        raise ValueError(f'Invalid split name: {row["split"]} and time_bin: {row["time_bin"]}')

df_case_good['final_split'] = df_case_good.apply(map_split_name, axis=1)
df_case_good['final_split'].value_counts().sort_index()

path = os.path.join(SPACE['DATA_SPLIT'], 'OhioT1DM_ds_case_split.parquet')
df_case_good.to_parquet(path)


print(df_case_good['final_split'].value_counts().sort_index())
print('save OhioT1DM_ds_case_split.parquet to', path)


'''
python 1-OhioT1DM/3_run_ds_case_split.py
'''