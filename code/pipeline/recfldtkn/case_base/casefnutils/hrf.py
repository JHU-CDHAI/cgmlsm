import datasets
import pandas as pd
import logging 


logger = logging.getLogger(__name__)
# move to HumanDirectoryArgs


def get_HumanDirectoryArgs_ForOneCase(case_example, HumanRecordRecfeat_Args):
    HumanID_list = [HumanName + 'ID' for HumanName in HumanRecordRecfeat_Args]
    HumanDirectory_Args = {}
    for HumanID in HumanID_list: 
        HumanName = HumanID.replace('ID', '')
        OneHumanDirectory_Args = {}
        HumanIDValueList = [case_example[HumanID]]
        OneHumanDirectory_Args['HumanName'] = HumanName
        OneHumanDirectory_Args['HumanIDValueList'] = HumanIDValueList
        HumanDirectory_Args[HumanName] = OneHumanDirectory_Args
    # HRFDirectory = get_HRFDirectory_from_HumanDirectory(onecohort_record_base, HumanDirectory_Args, HumanRecordRecfeat_Args)
    return HumanDirectory_Args



def get_HumanDirectoryArgs_ForBatch(df_case, HumanRecordRecfeat_Args):
    # df_case = pd.DataFrame(case_examples)
    # print(df_case)
    HumanID_list = [HumanName + 'ID' for HumanName in HumanRecordRecfeat_Args]
    HumanDirectory_Args = {}
    for HumanID in HumanID_list: 
        HumanName = HumanID.replace('ID', '')
        OneHumanDirectory_Args = {}
        HumanIDValueList = df_case[HumanID].unique().tolist()
        OneHumanDirectory_Args['HumanName'] = HumanName
        OneHumanDirectory_Args['HumanIDValueList'] = HumanIDValueList
        HumanDirectory_Args[HumanName] = OneHumanDirectory_Args


    # HRFDirectory = get_HRFDirectory_from_HumanDirectory(onecohort_record_base, HumanDirectory_Args, HumanRecordRecfeat_Args)
    return HumanDirectory_Args



def get_RecAndRecFeat_from_recordbase(onecohort_record_base, 
                                      HumanID, 
                                      HumanName,
                                      record, 
                                      RecordName,
                                      RecordFeat_List,
                                      ):
    
    Record_Proc_Config = onecohort_record_base.Record_Proc_Config
    if Record_Proc_Config is None: Record_Proc_Config = {}
    via_method = Record_Proc_Config.get('via_method', 'ds')

    
    # if via_method == 'ds': #  hasattr(record, 'ds_RecAttr'):
        
    #     # during the training process
    #     if record is None:
    #         # the key problem is ds_RecAttr cannot be empty with column names only.
    #         ds_RecAttr = None
    #         ds_RecIndex = None
    #     else:
    #         # assert len(ds_RecAttr) > 0
    #         # if len(ds_RecAttr) == 0:
    #         #     # print('ds_RecAttr is empty')
    #         #     logger.warning(f'ds_RecAttr is empty for {HumanName} -- {RecordName}') 
    #         ds_RecAttr = record.ds_RecAttr 

    #         ############################################ # added by jluo: 2025-05-04
    #         # always using the df now. 
    #         ds_RecAttr = ds_RecAttr.to_pandas()
    #         ############################################

    #         ds_RecIndex = record.ds_RecIndex

    #     if ds_RecAttr is None:
    #         # logger.warning(f'ds_RecAttr is empty for {HumanName} -- {RecordName}')   
    #         df_RecIndex = pd.DataFrame(columns = [HumanID]).set_index(HumanID)
    #     else:
    #         df_RecIndex = ds_RecIndex.to_pandas().set_index(HumanID)

    # elif via_method == 'df':

    #     #################################
    #     # for the inference process.
    #     assert record is not None

    #     # if record is not available for a patient, then record.df_RecAttr is an empty dataframe with columns. 
    #     #################################

    #     # ds_RecAttr = datasets.Dataset.from_pandas(record.df_RecAttr)
    #     ds_RecAttr = record.df_RecAttr # could be df or ds
    #     # print(record.df_RecIndex)
    #     df_RecIndex = record.df_RecIndex.set_index(HumanID)
    #     if len(record.df_RecAttr) == 0:
    #         # print('ds_RecAttr is empty')
    #         logger.warning(f'ds_RecAttr is empty for {HumanName} -- {RecordName}')
            
    # else:
    #     raise ValueError(f'via_method: {via_method} is not supported for RecAttr')


    #################################
    # for the inference process.
    assert record is not None

    # if record is not available for a patient, then record.df_RecAttr is an empty dataframe with columns. 
    #################################


    if hasattr(record, 'df_RecAttr'):
        logger.info('record: get df_RecAttr from df_RecAttr')
        df_RecAttr = record.df_RecAttr
    else:
        logger.info('record: get df_RecAttr from ds_RecAttr.to_pandas()')
        df_RecAttr = record.ds_RecAttr.to_pandas()
        record.df_RecAttr = df_RecAttr

    # ds_RecAttr = datasets.Dataset.from_pandas(record.df_RecAttr)
    # ds_RecAttr = record.ds_RecAttr # could be df or ds

    # print(record.df_RecIndex)
    df_RecIndex = record.df_RecIndex# .set_index(HumanID)
    if len(record.df_RecAttr) == 0:
        # print('ds_RecAttr is empty')
        logger.warning(f'ds_RecAttr is empty for {HumanName} -- {RecordName}')


    # recfeat_list = [onecohort_record_base.Name_to_HRF[(HumanName, RecordName, RecFeatName)] for RecFeatName in RecordFeat_List]
    
    # RecFeatName_to_dsRecFeat = {}
    # for rf in recfeat_list:
    #     if via_method == 'ds':
    #         RecFeatName_to_dsRecFeat[rf.RecFeatName] = rf.ds_RecFeat
    #     elif via_method == 'df':
    #         RecFeatName_to_dsRecFeat[rf.RecFeatName] = rf.df_RecFeat # datasets.Dataset.from_pandas()
    #         # logger.info('using "df" mode in get_HRFDirectory_from_HumanDirectory')
    #     else:
    #         raise ValueError(f'via_method: {via_method} is not supported for RecFeat')

    results = {
        'df_RecAttr': df_RecAttr,
        'df_RecIndex': df_RecIndex,
        # 'RecFeatName_to_dsRecFeat': RecFeatName_to_dsRecFeat
    }
    return results 


def get_HRFDirectory_from_HumanDirectory(onecohort_record_base, HumanDirectory_Args, HumanRecordRecfeat_Args):
    # print(HumanRecordRecfeat_Args)
    HRFDirectory = {}
    for HumanName, OneHumanDirectoryArgs in HumanDirectory_Args.items():
        # prepare the Human Directory Information
        human = onecohort_record_base.Name_to_HRF[HumanName]
        HumanID = human.HumanID
        HumanIDValueList =  OneHumanDirectoryArgs['HumanIDValueList']
        # --------- this part could be optimized to Multi-Threading or Multi-Processing
        for HumanIDValue in HumanIDValueList: 
            HRFDirectory[(HumanName, HumanIDValue)] = {}
        
        RecordRecFeat_Args = HumanRecordRecfeat_Args[HumanName]
        for RecordName, RecordFeat_List in RecordRecFeat_Args.items():
            
            # print(RecordName, RecordFeat_List)
            # prepare the record
            # record = onecohort_record_base.Name_to_HRF[(HumanName, RecordName)]
            record = onecohort_record_base.Name_to_HRF.get((HumanName, RecordName), None)

            # if record is None:
            #     logger.warning(f'record is None for {HumanName} -- {RecordName}')
            #     continue
            
            if not hasattr(record, 'df_RecIndex'):
                df_RecIndex = record.ds_RecIndex.to_pandas().set_index(HumanID)
                record.df_RecIndex = df_RecIndex
            df_RecIndex = record.df_RecIndex


            if record.shadow_df == True:
                if not hasattr(record, 'df_RecAttr'):
                    df_RecAttr = record.ds_RecAttr.to_pandas()
                    record.df_RecAttr = df_RecAttr
                data_RecAttr = record.df_RecAttr
            else:
                data_RecAttr = record.ds_RecAttr
            
            # RecFeatName_to_dsRecFeat = results['RecFeatName_to_dsRecFeat']

            # print(RecFeatName_to_dsRecFeat)
            for HumanIDValue in HumanIDValueList:
                # logger.info(f'{HumanIDValue} -- {df_RecIndex.index}')
                RFInfo = HRFDirectory[(HumanName, HumanIDValue)] 
                
                if HumanIDValue not in df_RecIndex.index: 
                    RFInfo[RecordName] = None
                    # for RecFeatName, ds_RecFeat in RecFeatName_to_dsRecFeat.items():
                    #     RFInfo[(RecordName, RecFeatName)] = None
                else:

                    # Record_Proc_Config = onecohort_record_base.Record_Proc_Config
                    # via_method = Record_Proc_Config.get('via_method', 'ds')
                    # if interval exists, ds_RecAttr or df_RecAttr must larger than 0.
                    
                    interval = df_RecIndex.loc[HumanIDValue]['interval']
                    # print(RecordName)
                    if type(data_RecAttr) == datasets.Dataset:
                        RFInfo[RecordName] = data_RecAttr.select(range(interval[0], interval[1] + 1)).to_pandas()
                    else:
                        # do not use .copy() otherwise it will take longer time. 
                        RFInfo[RecordName] = data_RecAttr.iloc[interval[0]:interval[1] + 1]# .copy()

                    # RFInfo[RecordName] = df_RecAttr.iloc[interval[0]:interval[1] + 1]# .copy()
                    # Record Feat
                    # for RecFeatName, ds_RecFeat in RecFeatName_to_dsRecFeat.items():
                    #     if type(ds_RecFeat) == datasets.Dataset:
                    #         RFInfo[(RecordName, RecFeatName)] = ds_RecFeat.select(range(interval[0], interval[1] + 1))
                    #     else:
                    #         RFInfo[(RecordName, RecFeatName)] = ds_RecFeat.iloc[interval[0]:interval[1] + 1]
                    
                    # Dates
                    if 'dates' in df_RecIndex.columns:
                        RFInfo[(RecordName, 'dates')] =  df_RecIndex.loc[HumanIDValue]['dates']

                # HRFDirectory[(HumanName, HumanIDValue)] = RFInfo

    return HRFDirectory
