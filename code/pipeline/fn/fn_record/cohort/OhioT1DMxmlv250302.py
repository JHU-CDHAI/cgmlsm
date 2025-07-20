import os

import pandas as pd

import numpy as np

OneCohort_Args = {'CohortLabel': 11,
 'CohortName': 'OhioT1DM',
 'SourcePath': './_Data/0-Data_Raw/OhioT1DM/Source',
 'FolderPath': './_Data/0-Data_Raw/OhioT1DM/',
 'Source2CohortName': 'OhioT1DMxmlv250302'}

SourceFile_SuffixList = ['xml']

def convert_ohio_xml_to_dataframes(data, xml_path=None):
    """
    Convert OhioT1DM XML data into pandas DataFrames for each section.

    Args:
        data (dict): The parsed XML data from OhioT1DM dataset
        xml_path (str, optional): Path to the XML file, used to extract year and dataset type

    Returns:
        dict: Dictionary mapping section names to pandas DataFrames
    """
    # import xmltodict


    dataframes = {}

    if 'patient' not in data:
        print("No patient data found in the XML")
        return dataframes

    patient = data['patient']
    patient_id = patient.get('@id', 'unknown')

    # Extract patient attributes
    patient_attrs = {k.replace('@', ''): v for k, v in patient.items() if k.startswith('@')}

    # Add year and dataset type (test/train) information from the file path if available
    if xml_path:
        # Extract year from the path using regex pattern matching
        import re
        year_match = re.search(r'/(\d{4})/', xml_path)
        if year_match:
            patient_attrs['year'] = year_match.group(1)
        else:
            patient_attrs['year'] = 'unknown'

        # Extract dataset type (test or train)
        if '/test/' in xml_path or '-testing' in xml_path:
            patient_attrs['dataset_type'] = 'test'
            patient_id = str(patient_id) + '_test'
        elif '/train/' in xml_path or '-training' in xml_path:
            patient_attrs['dataset_type'] = 'train'
            patient_id = str(patient_id) + '_train'
        else:
            patient_attrs['dataset_type'] = 'unknown'


        patient_attrs['patient_id'] = patient_id
        # Add the full file path for reference
        patient_attrs['file_path'] = xml_path

    patient_df = pd.DataFrame([patient_attrs])
    dataframes['patient_info'] = patient_df

    # Process each section
    for section_name, section_data in patient.items():
        if section_name.startswith('@') or not isinstance(section_data, dict):
            continue

        if 'event' in section_data:
            events = section_data['event']
            if not isinstance(events, list):
                events = [events]  # Convert single event to list

            # Extract all events into a list of dictionaries
            events_list = []
            for event in events:
                event_dict = {k.replace('@', ''): v for k, v in event.items()}
                event_dict['patient_id'] = patient_id  # Add patient ID to each event
                events_list.append(event_dict)

            # Create DataFrame from events
            if events_list:
                section_df = pd.DataFrame(events_list)

                # Convert datetime-like columns to datetime
                datetime_columns = ['ts', 'tbegin', 'tend']
                for col in datetime_columns:
                    if col in section_df.columns:
                        section_df[col] = pd.to_datetime(section_df[col], format='%d-%m-%Y %H:%M:%S', errors='coerce')

                dataframes[section_name] = section_df

    return dataframes


def get_RawName_from_SourceFile(file_path, OneCohort_Args):
    """
    This one is useless 
    """
    RawName = file_path.split('_')[-1].split('.')[0]
    return RawName


def process_Source_to_Raw(OneCohort_Args, SourceFile_List, get_RawName_from_SourceFile,SPACE):
    """
    Process source files to raw data files, including renaming columns and merging certain files.

    Args:
    OneCohort_Args (dict): Dictionary containing processing arguments, including 'FolderPath'.
    SourceFile_List (list): List of source file paths.
    get_RawName_from_SourceFile (function): Function to extract raw name from file path.

    Returns:
    dict: Mapping of raw names to processed file paths.
    """

    from collections import defaultdict


    # Initialize dictionary to store raw names and their corresponding file paths
    RawName_to_dfRaw = {}
    # for file_path in SourceFile_List:
    #     # Extract the raw name for each file using the function 
    #     RawName = get_RawName_from_SourceFile(file_path, OneCohort_Args)
    #     # Assign value file_path to key RawName
    #     RawName_to_dfRaw[RawName] = file_path

    # import xmltodict

    combined_dfs = defaultdict(list)

    # print("Processing all XML files in the OhioT1DM dataset...")
    for xml_file in SourceFile_List:
        try:
            # Parse XML file
            # print(f"Processing file: {os.path.basename(xml_file)}")
            with open(xml_file, 'r') as f:
                xml_content = f.read()

            # Convert XML to dictionary using xmltodict
            import xmltodict
            data = xmltodict.parse(xml_content)

            # Convert to DataFrames
            if 'patient' in data:
                patient_dfs = convert_ohio_xml_to_dataframes(data, xml_path=xml_file)

                # Add each DataFrame to the combined collection
                for section_name, df in patient_dfs.items():
                    if not df.empty:
                        combined_dfs[section_name].append(df)
        except Exception as e:
            print(f"Error processing {xml_file}: {str(e)}")



    final_dfs = {}
    for section_name, df_list in combined_dfs.items():
        if df_list:
            final_dfs[section_name] = pd.concat(df_list, ignore_index=True)


    # ---------- process patient_info --> Patient
    df = final_dfs['patient_info']
    RawName = 'Ptt'
    raw_columns = ['PatientID', 'MRSegmentID', 'MRSegmentModifiedDateTime', 'DiseaseType',
                                'Gender', 'ActivationDate', 'UserTimeZoneOffset', 'UserTimeZone',
                                'Description', 'YearOfBirth']

    df = df.rename(columns = {'patient_id': 'PatientID'})
    df = df.reindex(columns = raw_columns)
    # df['BGEntryID'] = df.index
    df['DiseaseType'] = 1
    df['UserTimeZoneOffset'] = 0
    RawName_to_dfRaw[RawName] = df 



    # ---------- process exercise --> ELogExerciseEntry
    df = final_dfs['exercise']
    # print(df['type'].value_counts())

    RawName = 'Exercise'

    raw_columns = ['ExerciseEntryID', 'PatientID',  # 'EntryID',
                'ExerciseDuration', 'ExerciseType', 'ExerciseIntensity',
                'TimeSinceExercise', #  'EntrySourceID', 
                # 'ActivityTypeID',
                'ObservationDateTime', 'ObservationEntryDateTime',
                'TimezoneOffset', 'Timezone', # 'EntryCreatedDateTime',
                # 'ObservationCreatedBy', 'ObservationStatus',
                # 'SourceReferenceID', 'ModifiedDateTime', 
                'CaloriesBurned',
                'DistanceInMeters', # 'ExternalEntryID', 'ExternalSourceID',
                'ExerciseType',
                ]

    df = df.rename(columns = {
        'patient_id': 'PatientID', 
        'carbs': 'CarbsValue',
        'ts': 'ObservationDateTime',
        'intensity': 'ExerciseIntensity', 
        'duration': 'ExerciseDuration',
    })

    df = df.reindex(columns = raw_columns)
    df['ExerciseEntryID'] = df.index
    # df['DiseaseType'] = 1
    df['TimezoneOffset'] = 0
    df['ObservationEntryDateTime'] = pd.to_datetime(df['ObservationDateTime'])
    df['ExerciseType'] = 'Unknown'
    # df.head()
    RawName_to_dfRaw[RawName] = df 



    # ---------- process meal --> ELogCarbsEntry
    # df = final_dfs['patient_info']

    ######### deal with the ElogBGEntry
    df = final_dfs['meal']


    RawName = 'Diet'
    raw_columns = ['PatientID', 'CarbsEntryID', # 'EntryID', 
                   'Carbs',
                    # 'EntrySourceID', 'ActivityTypeID', 
                    'ObservationDateTime',
                    # 'ObservationEntryDateTime',
                    'TimezoneOffset', 'Timezone',
                    # 'EntryCreatedDateTime', 'ObservationCreatedBy',
                    # 'ObservationStatus', 'SourceReferenceID', 'ModifiedDateTime',
                    'ExternalSourceID', 'ExternalEntryID', 'TotalCalories'
                    ]

    df = df.rename(columns = {
        'patient_id': 'PatientID', 
        'carbs': 'Carbs',
        'ts': 'ObservationDateTime',
    })

    df = df.reindex(columns = raw_columns)
    df['CarbsEntryID'] = df.index
    # df['DiseaseType'] = 1
    df['TimezoneOffset'] = 0
    df['ObservationEntryDateTime'] = pd.to_datetime(df['ObservationDateTime'])
    df['FoodName'] = 'Unknown'
    # df.head()
    RawName_to_dfRaw[RawName] = df 


    # ---------- process glucose_level --> ElogBGEntry
    df = final_dfs['glucose_level']
    RawName = 'ElogBGEntry'
    raw_columns = ['BGEntryID', 'PatientID', 'ObservationDateTime', 'BGValue',
                                    # 'IsNormalIndicator', 'ObservationEntryDateTime', 
                                    'TimezoneOffset',
                                    'Timezone', # 'EntryCreatedDateTime', 'ActualBGValue',
                                    'EntrySourceID', # 'UserObservationDateTime'
                                    ]
    df = df.rename(columns = {'patient_id': 'PatientID', 'value': 'BGValue', 'ts': 'ObservationDateTime'})
    df = df.reindex(columns = raw_columns)
    df['BGEntryID'] = df.index
    df['TimezoneOffset'] = 0 
    df['EntrySourceID'] = 18
    # df['EntryCreatedDateTime'] = pd.to_datetime(df['ObservationDateTime'])
    RawName_to_dfRaw[RawName] = df

    df['BGValue'] = pd.to_numeric(df['BGValue'], errors='coerce')
    df = df[df['BGValue'].notna()].reset_index(drop = True)
    df['BGValue'] = df['BGValue'].astype(int)

    for RawName, df in RawName_to_dfRaw.items():
        print(RawName, df.shape)
        print(df.columns)
        # display(df.head())

        path = os.path.join(OneCohort_Args['FolderPath'], f'processed_RawFile_{RawName}.csv')
        df.to_csv(path, index=False)
        RawName_to_dfRaw[RawName] = path# .replace(SPACE['DATA_RAW'], '$DATA_RAW$')

    return RawName_to_dfRaw


MetaDict = {
	"OneCohort_Args": OneCohort_Args,
	"SourceFile_SuffixList": SourceFile_SuffixList,
	"convert_ohio_xml_to_dataframes": convert_ohio_xml_to_dataframes,
	"get_RawName_from_SourceFile": get_RawName_from_SourceFile,
	"process_Source_to_Raw": process_Source_to_Raw
}