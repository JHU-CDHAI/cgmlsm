import pandas as pd

import numpy as np

OneRecord_Args = {'RecordName': 'Med5Min',
 'RecID': 'Med5MinID',
 'RecIDChain': ['PID'],
 'RawHumanID': 'PatientID',
 'ParentRecName': 'P',
 'RecDT': 'DT_s',
 'RawNameList': ['MedAdmin'],
 'human_group_size': 100,
 'rec_chunk_size': 100000}

RawName_to_RawConfig = {'MedAdmin': {'raw_columns': ['PatientID', 'EntryDateTime', 'CreatedDate',
                              'AdministrationTimeZoneOffset', 'AdministrationTimeZone',
                              'AdministrationDate', 'ActivityTypeID', 'Dose', 'MedSourceID',
                              'MedicationID', 'AdministrationTimeLabelID', 'ModifiedDateTime',
                              'UserAdministrationDate'],
              'rec_chunk_size': 100000,
              'raw_datetime_column': 'EntryDateTime'}}

attr_cols = ['PID', 'PatientID', 'Med5MinID', 'DT_tz', 'DT_r', 'DT_s', 'MedicationID', 'Dose', 'time_to_last_entry']

def get_RawRecProc_for_HumanGroup(df_RawRec_for_HumanGroup, OneRecord_Args, df_Human):
    df = df_RawRec_for_HumanGroup

    # 1. filter out the records we don't need (optional) 
    df = df[df['AdministrationTimeZoneOffset'].abs() < 1000].reset_index(drop = True)

    # 2. entry type

    # 3. update datetime columns 
    DTCol_list = ['AdministrationDate', 
                  'EntryDateTime', 
                #   'CreatedDate',
                #   'MedPrescriptionTime', 
                #   'ModifiedDateTime',
                'UserAdministrationDate'
                  ]
    for DTCol in DTCol_list: 
        df[DTCol] = pd.to_datetime(df[DTCol], format = 'mixed')

    # x1. localize the datetime columns to based on time zone. 
    a = len(df)
    df = pd.merge(df, df_Human[['PatientID', 'user_tz']],  how = 'left')
    b = len(df)
    assert a == b

    # df['DT_tz'] = df['AdministrationTimeZoneOffset'].replace(0, None).fillna(df['user_tz'])   
    df['DT_tz'] = df['AdministrationTimeZoneOffset'].replace(0, None).infer_objects().fillna(df['user_tz']).fillna(0).astype(int)  

    DTCol = 'DT_r'
    DTCol_source = 'EntryDateTime'
    df[DTCol] = df[DTCol_source]
    df[DTCol] = pd.to_datetime(df[DTCol]) + pd.to_timedelta(df['DT_tz'], 'm')
    assert df[DTCol].isna().sum() == 0


    DTCol = 'DT_s'
    DTCol_source = 'AdministrationDate'
    # DTCol_source = 'UserAdministrationDate'
    df[DTCol] = df[DTCol_source]
    df[DTCol] = pd.to_datetime(df[DTCol]).apply(lambda x: None if x <= pd.to_datetime('2010-01-01') else x)
    df[DTCol] = pd.to_datetime(df[DTCol]) + pd.to_timedelta(df['DT_tz'], 'm')
    df[DTCol] = df[DTCol].fillna(df['DT_r'])
    assert df[DTCol].isna().sum() == 0

    # DTCol = 'DT_e'
    # DTCol_source = None
    # df[DTCol] = df[DTCol_source] if DTCol_source in df.columns else pd.to_datetime([None] * len(df))
    # df[DTCol] = pd.to_datetime(df[DTCol]).apply(lambda x: None if x <= pd.to_datetime('2010-01-01') else x)
    # df[DTCol] = pd.to_datetime(df[DTCol]) + pd.to_timedelta(df['DT_tz'], 'm')
    # df[DTCol] = df[DTCol].fillna(df['DT_s'])
    # assert df[DTCol].isna().sum() == 0

    # x3. drop duplicates


    def densify_timestamps(df):
        time_interval = pd.Timedelta(minutes=31)
        df_sorted = df.sort_values(by=['PatientID', 'DT_s']).copy()  # Ensure sorting

        def adjust_group(group):
            timestamps = group['DT_s'].tolist()
            updated_timestamps = timestamps.copy()  # Preserve original order
            i = 0

            while i < len(timestamps):
                t1 = timestamps[i]
                j = i + 1  # Start checking from the next timestamp

                # Adjust following timestamps if they fall within the 15-minute window
                while j < len(timestamps) and timestamps[j] <= t1 + time_interval:
                    updated_timestamps[j] = t1  # Set to t1
                    j += 1

                i = j  # Move to the next unprocessed timestamp

            group['DT_s'] = updated_timestamps
            return group

        # Apply to each patient group separately
        columns = df_sorted.columns.tolist()
        columns = [i for i in columns if i != 'PatientID']
        df_updated = df_sorted.groupby('PatientID')[columns].apply(adjust_group)

        return df_updated

    ###### densify timestamps to 30 minutes. 
    df = densify_timestamps(df)



    ###### update the format to per 5-minutes.
    DTCol_list = ['DT_s', 
                  'DT_r', 
                  # 'DT_e',
                  ] # 
    for DTCol in DTCol_list:
        # DateTimeUnit ='5Min'
        date = df[DTCol].dt.date.astype(str)
        hour = df[DTCol].dt.hour.astype(str)
        minutes = ((df[DTCol].dt.minute / 5).astype(int) * 5).astype(str)
        df[DTCol] = pd.to_datetime(date + ' ' + hour +':' + minutes + ':' + '00')

    df = df.drop_duplicates()


    # 4. select a DT as the RecDT
    # RecDT = 'DT_s'

    RawHumanID = OneRecord_Args['RawHumanID']
    RecDT = 'DT_s'
    df = df.groupby([RawHumanID, RecDT]).agg(
        {
            # 'PatientID': 'first',
            'DT_r': 'first',
            'DT_tz': 'first',

            "MedicationID": lambda x: "; ".join(str(i) for i in x),
            "Dose": lambda x: "; ".join(str(i) for i in x),
            # "ActivityTypeID": lambda x: "; ".join(str(i) for i in x),
        }
    ).reset_index()
    df['time_to_last_entry'] = df.groupby('PatientID', group_keys=False)['DT_s'].diff().dt.total_seconds() / 60 / 5
    # ----------------------------------------------------------------- #



    df_RawRecProc = df
    return df_RawRecProc 


MetaDict = {
	"OneRecord_Args": OneRecord_Args,
	"RawName_to_RawConfig": RawName_to_RawConfig,
	"attr_cols": attr_cols,
	"get_RawRecProc_for_HumanGroup": get_RawRecProc_for_HumanGroup
}