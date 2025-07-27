import pandas as pd

import numpy as np

OneRecord_Args = {'RecordName': 'Exercise5Min',
 'RecID': 'Exercise5MinID',
 'RecIDChain': ['PID'],
 'RawHumanID': 'PatientID',
 'ParentRecName': 'P',
 'RecDT': 'DT_s',
 'RawNameList': ['Exercise'],
 'human_group_size': 100,
 'rec_chunk_size': 100000}

RawName_to_RawConfig = {'Exercise': {'raw_columns': ['PatientID', 'ObservationDateTime', 'ObservationEntryDateTime',
                              'TimezoneOffset', 'Timezone', 'ExerciseType', 'ExerciseIntensity',
                              'TimeSinceExercise', 'ActivityTypeID', 'ExerciseDuration',
                              'CaloriesBurned', 'DistanceInMeters', 'ActivityType'],
              'rec_chunk_size': 100000}}

attr_cols = ['PID', 'PatientID', 'Exercise5MinID', 'DT_tz', 'DT_r', 'DT_s', 'ExerciseDuration', 'ExerciseIntensity', 'CaloriesBurned', 'DistanceInMeters', 'ExerciseType', 'time_to_last_entry']

def get_RawRecProc_for_HumanGroup(df_RawRec_for_HumanGroup, OneRecord_Args, df_Human):
    df = df_RawRec_for_HumanGroup

    # 1. filter out the records we don't need (optional) 
    df = df[df['TimezoneOffset'].abs() < 1000].reset_index(drop = True)

    # 2. entry type

    # 3. update datetime columns 
    DTCol_list = [
        'ObservationDateTime', 
        'ObservationEntryDateTime',
        # 'EntryCreatedDateTime', 
        # 'ModifiedDateTime',
    ]

    for DTCol in DTCol_list: 
        df[DTCol] = pd.to_datetime(df[DTCol], format = 'mixed')

    # x1. localize the datetime columns to based on time zone. 
    a = len(df)
    df = pd.merge(df, df_Human[['PatientID', 'user_tz']],  how = 'left')
    b = len(df)
    assert a == b
    df['DT_tz'] = df['TimezoneOffset'].replace(0, None).fillna(df['user_tz']).infer_objects(copy=False)


    DTCol = 'DT_r'
    DTCol_source = 'ObservationEntryDateTime'
    df[DTCol] = df[DTCol_source]
    df[DTCol] = pd.to_datetime(df[DTCol]) + pd.to_timedelta(df['DT_tz'], 'm')
    assert df[DTCol].isna().sum() == 0

    DTCol = 'DT_s'
    DTCol_source = 'ObservationDateTime'
    df[DTCol] = df[DTCol_source]
    df[DTCol] = pd.to_datetime(df[DTCol]).apply(lambda x: None if x <= pd.to_datetime('2010-01-01') else x)
    df[DTCol] = pd.to_datetime(df[DTCol]) + pd.to_timedelta(df['DT_tz'], 'm')
    df[DTCol] = df[DTCol].fillna(df['DT_r'])
    assert df[DTCol].isna().sum() == 0

    DTCol = 'DT_e'
    DTCol_source = None
    # select
    df['DT_e'] = df['DT_s'] + pd.to_timedelta(df['ExerciseDuration'], 'm')
    assert df[DTCol].isna().sum() == 0

    # # x3. drop duplicates
    df = df.drop_duplicates()

    df['DT_tz'] = df['DT_tz'].fillna(0).astype(int)




    # 4. select a DT as the RecDT
    # RecDT = 'DT_s'

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


    df = densify_timestamps(df)

    DTCol_list = ['DT_s', 
                  'DT_r', 
                  'DT_e',
                  ] # 
    for DTCol in DTCol_list:
        # DateTimeUnit ='5Min'
        date = df[DTCol].dt.date.astype(str)
        hour = df[DTCol].dt.hour.astype(str)
        minutes = ((df[DTCol].dt.minute / 5).astype(int) * 5).astype(str)
        df[DTCol] = pd.to_datetime(date + ' ' + hour +':' + minutes + ':' + '00')

    df = df.drop_duplicates()
    df['ExerciseDuration'] = df['ExerciseDuration'].astype(float)

    RawHumanID = OneRecord_Args['RawHumanID']
    RecDT = 'DT_s'
    df = df.groupby([RawHumanID, RecDT]).agg(
        {
            # 'PatientID': 'first',
            'DT_r': 'first',
            'DT_tz': 'first',

            "ExerciseType": lambda x: "; ".join(x),
            "ExerciseIntensity": "first",
            'ExerciseDuration': 'sum', # should this be sum?
            "CaloriesBurned": "sum",
            'DistanceInMeters': 'sum',
            "ActivityType": "first",
        }
    ).reset_index()
    df['time_to_last_entry'] = df.groupby('PatientID', group_keys=False)['DT_s'].diff().dt.total_seconds() / 60 / 5
    # ----------------------------------------------------------------- #

    # drop the ExerciseDuration > 1000
    df = df[df['ExerciseDuration'] <= 120].reset_index(drop=True)
    df = df[df['ExerciseDuration'] >=5].reset_index(drop=True)

    df_RawRecProc = df
    return df_RawRecProc 


MetaDict = {
	"OneRecord_Args": OneRecord_Args,
	"RawName_to_RawConfig": RawName_to_RawConfig,
	"attr_cols": attr_cols,
	"get_RawRecProc_for_HumanGroup": get_RawRecProc_for_HumanGroup
}