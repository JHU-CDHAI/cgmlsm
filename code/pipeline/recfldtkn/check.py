import os
import itertools 
from .case_base.caseset import get_CaseFnTaskArgs_from_CaseFnNameList



def retrive_pipeline_info(SPACE):
    fn_path = os.path.join(SPACE['CODE_FN'], 'fn', 'fn_record')
    RFT_TO_FN = {}
    rft_list = ['cohort', 'human', 'record']
    for rft in rft_list:
        fn_data_type_path = os.path.join(fn_path, rft)    
        fn_list = [i.split('.')[0] for i in os.listdir(fn_data_type_path) if '.py' in i]
        fn_list = [i.replace('_', '-') for i in sorted(fn_list)]
        RFT_TO_FN[rft] = fn_list
        
    fn_path = os.path.join(SPACE['CODE_FN'], 'fn', 'fn_case')
    CASE_TO_FN = {}
    rft_list = ['trigger', 'case_casefn']
    for rft in rft_list:
        fn_data_type_path = os.path.join(fn_path, rft)    
        fn_list = [i.split('.')[0] for i in os.listdir(fn_data_type_path) if '.py' in i]
        CASE_TO_FN[rft] = sorted(fn_list)  
        
    fn_path = os.path.join(SPACE['CODE_FN'], 'fn', 'fn_aidata')
    AIDATA_TO_FN = {}
    rft_list = ['entryinput', 'entryoutput']
    for rft in rft_list:
        fn_data_type_path = os.path.join(fn_path, rft)    
        fn_list = [i.split('.')[0] for i in os.listdir(fn_data_type_path) if '.py' in i]
        # fn_list = [i.replace('_', '-') for i in sorted(fn_list)]
        AIDATA_TO_FN[rft] = fn_list
        
    PIPELINE_INFO = {
        'RFT_TO_FN': RFT_TO_FN,
        'CASE_TO_FN': CASE_TO_FN,
        'AIDATA_TO_FN': AIDATA_TO_FN,
    }
    return PIPELINE_INFO


def update_and_assert_CohortInfo(CohortName_to_OneCohortArgs, 
                                 CohortName_list,
                                 Record_Proc_Config, 
                                 Inference_Entry,
                                 ):
    CohortName_to_OneCohortArgs = {k: CohortName_to_OneCohortArgs[k] for k in CohortName_list}
    # check
    for cohort_name in CohortName_list:
        assert cohort_name in CohortName_to_OneCohortArgs, f'{cohort_name} not in CohortName_to_OneCohortArgs'
        
        
    CohortSettingInfo = {
        'CohortName_to_OneCohortArgs': CohortName_to_OneCohortArgs,
        'CohortName_list': CohortName_list,
        'Record_Proc_Config': Record_Proc_Config,
        'Inference_Entry': Inference_Entry
    }
    return CohortSettingInfo


def update_and_assert_CaseInfo(
                               TriggerCaseBaseName,
                               TriggerCaseBaseArgs,
                               Case_Args_Settings,
                               Case_Proc_Config = None, 
                               PIPELINE_INFO = None, 
                               SPACE = None,
                               ):
    
    CASE_TO_FN = PIPELINE_INFO['CASE_TO_FN']

    TriggerName = TriggerCaseBaseArgs['Trigger']['TriggerName']
    
    
    # TriggerCaseBaseName_to_TriggerCaseBaseArgs = {
    #     TriggerCaseBaseName: TriggerCaseBaseArgs
    # }

    # TriggerName = TriggerCaseBaseArgs['Trigger']['TriggerName']

    TagRec_list = [] 
    Filter_list = []
    Group_list = []
    CF_list = []
    # TagCF_list = []

    assert TriggerName in CASE_TO_FN['trigger'], f'{TriggerName} not in {CASE_TO_FN["trigger"]}'

    for k, args in TriggerCaseBaseArgs.items():
        # assert k in TriggerCaseBaseArgs, f'{k} not in TriggerCaseBaseArgs'
        TagRec_list += args.get('TagRec', [])
        Filter_list.append(args.get('Filter', ''))
        # ObsTask = args.get('ObsTask', {})
        CaseFnTasks = args.get('CaseFnTasks', [])
        CF_list += CaseFnTasks
    Filter_list = [i for i in Filter_list if i != '']
        
        
    ###### check the CF's availablility ####### 
    CF_list_total = CF_list
    CASE_TO_FN = PIPELINE_INFO['CASE_TO_FN']
    fn_list = CASE_TO_FN['case_casefn']
    for cf in CF_list_total:
        assert cf in fn_list, f'{cf} not in {fn_list}'

    # --------------------- assert ---------------------
    TagRec_to_TagRecArgs = Case_Args_Settings['TagRec_to_TagRecArgs']
    TagRec_to_TagRecArgs = {k: v for k, v in TagRec_to_TagRecArgs.items() if k in TagRec_list}
    for tag_rec in TagRec_list:
        assert tag_rec in TagRec_to_TagRecArgs, f'{tag_rec} not in TagRec_to_TagRecArgs'

    FltName_to_FltArgs = Case_Args_Settings['FltName_to_FltArgs']
    FltName_to_FltArgs = {k: v for k, v in FltName_to_FltArgs.items() if k in Filter_list}
    for flt in Filter_list:
        assert flt in FltName_to_FltArgs, f'{flt} not in FltName_to_FltArgs'
        
    
    CaseFnTaskArgs = get_CaseFnTaskArgs_from_CaseFnNameList(CF_list_total, SPACE = SPACE)
    HumanRecordRecfeat_Args = CaseFnTaskArgs['HumanRecordRecfeat_Args']
        
    RFT_TO_FN = PIPELINE_INFO['RFT_TO_FN']
    for Human, Rec2RecFeat in HumanRecordRecfeat_Args.items():
        assert Human in RFT_TO_FN['human']
        for Record, RecFeat_list in Rec2RecFeat.items():
            assert Record in RFT_TO_FN['record'], f'{Record} not in {RFT_TO_FN["record"]}'
            for RecFeat in RecFeat_list:
                # print(RecFeat)
                # print(RFT_TO_FN['recfeat'])
                assert RecFeat in RFT_TO_FN['recfeat'], f'{RecFeat} not in {RFT_TO_FN["recfeat"]}'
                
                
    TagRec_columns = list(itertools.chain(*[TagRec_to_TagRecArgs[TagRecName]['columns_to_tag'] for TagRecName in TagRec_list]))
    # TagCF_columns = list(itertools.chain(*[TagCF_to_TagCFArgs[TagCF]['tkn_name_list'] for TagCF in TagCF_list]))



    Case_Args_Settings = {
        'TagRec_to_TagRecArgs': TagRec_to_TagRecArgs,
        'FltName_to_FltArgs': FltName_to_FltArgs,
        # 'GROUP_TO_GROUPMethodArgs': GROUP_TO_GROUPMethodArgs,
        # 'Ckpd_to_CkpdObsConfig': Ckpd_to_CkpdObsConfig,
        # 'CF_to_CFArgs': CF_to_CFArgs,
        # 'TagCF_to_TagCFArgs': TagCF_to_TagCFArgs,
    }
    
    
    CohortSettingInfo = {
        
        'HumanRecordRecfeat_Args': HumanRecordRecfeat_Args,
        'Case_Args_Settings': Case_Args_Settings,
        'Case_Proc_Config': Case_Proc_Config,
        
        # 'TriggerName': TriggerName,
        # 'TriggerCaseBaseName_to_TriggerCaseBaseArgs': TriggerCaseBaseName_to_TriggerCaseBaseArgs,
        'TriggerCaseBaseName': TriggerCaseBaseName, 
        # 'TriggerCaseBaseName_to_TriggerCaseBaseArgs': TriggerCaseBaseName_to_TriggerCaseBaseArgs,
        'TriggerCaseBaseArgs': TriggerCaseBaseArgs,
        
       
        'TagRec_columns': TagRec_columns,
        # 'TagCF_columns': TagCF_columns,
        'CF_list': CF_list,
    }
    
    
    return CohortSettingInfo


def update_and_assert_AIDataInfo(
        OneAIDataName_to_OneAIDataArgs,
        AIDataArgs_columns,
        AIDATA_TO_FN, 
    ):

    ############ Input Part ############
    # Input_Args = AIData_Job_Args['Input_Args']
    # InputName_to_Settings = AIDataSettings['InputName_to_Settings']
    # InputName_to_Settings['INPUT_CFs_Args'] = {k: v for k, v in InputName_to_Settings['INPUT_CFs_Args'].items() if k == Input_Args['INPUT_CFs_Args_Name']}
    # INPUT_CFs_Args_Name = Input_Args['INPUT_CFs_Args_Name']
    # assert INPUT_CFs_Args_Name in InputName_to_Settings['INPUT_CFs_Args']

    # EntryInputMethod = Input_Args['EntryInputMethod']
    # assert EntryInputMethod in AIDATA_TO_FN['entryinput']
    EntryInputMethod_list = [v['OneEntryArgs']['Input_Part']['EntryInputMethod'] for k, v in OneAIDataName_to_OneAIDataArgs.items()]
    for EntryInputMethod in EntryInputMethod_list:
        assert EntryInputMethod in AIDATA_TO_FN['entryinput'], f'{EntryInputMethod} not in {AIDATA_TO_FN["entryinput"]}'


    ############ Output Part ############
    # Tasks_Series_Args = AIData_Job_Args['Tasks_Series_Args']

    # TaskType = Tasks_Series_Args['TaskType']

    # TaskType_to_TaskSeriesNameList = {k: v for k, v in TaskType_to_TaskSeriesNameList.items() if k == TaskType}
    # TaskType_to_EntryOutputMethod  = {k: v for k, v in TaskType_to_EntryOutputMethod.items()  if k == TaskType}
    # assert TaskType in TaskType_to_TaskSeriesNameList
    # assert TaskType in TaskType_to_EntryOutputMethod


    EntryOutputMethod_list = [v['OneEntryArgs']['Output_Part']['EntryOutputMethod'] for k, v in OneAIDataName_to_OneAIDataArgs.items()]
    for EntryOutputMethod in EntryOutputMethod_list:
        assert EntryOutputMethod in AIDATA_TO_FN['entryoutput']


    # EntryOutputMethod = Tasks_Series_Args['EntryOutputMethod']
    # assert EntryOutputMethod in AIDATA_TO_FN['entryoutput']

    # TaskSeriesName_List = Tasks_Series_Args['TaskSeriesName_List']
    
    # TasksName_to_Settings = AIDataSettings['TasksName_to_Settings']
    # TasksName_to_Settings = {k: v for k, v in TasksName_to_Settings.items() if k in TaskSeriesName_List}
    # TaskType_to_TaskSeriesNameList[TaskType] = [i for i in TaskType_to_TaskSeriesNameList[TaskType] if i in TaskSeriesName_List]
    # for TaskSeriesName in TaskSeriesName_List:
    #     assert TaskSeriesName in TaskType_to_TaskSeriesNameList[TaskType]
        
        
    # AIDev_Args = AIData_Job_Args['AIDev_Args']
    # AIDevName_to_Settings = AIDataSettings['AIDevName_to_Settings']

    # NewName_to_OldNames_Name = AIDev_Args['NewName_to_OldNames_Name']
    # AIDevName_to_Settings['NewName_to_OldNames'] = {k: v for k, v in AIDevName_to_Settings['NewName_to_OldNames'].items() if k == NewName_to_OldNames_Name}
    # assert NewName_to_OldNames_Name in AIDevName_to_Settings['NewName_to_OldNames']

    # TrainEvals_Name = AIDev_Args['TrainEvals_Name']
    # AIDevName_to_Settings['TrainEvals'] = {k: v for k, v in AIDevName_to_Settings['TrainEvals'].items() if k == TrainEvals_Name}
    # assert TrainEvals_Name in AIDevName_to_Settings['TrainEvals']

    # SplitTagging_Name = AIDev_Args['SplitTagging_Name']
    # AIDevName_to_Settings['SplitTagging'] = {k: v for k, v in AIDevName_to_Settings['SplitTagging'].items() if k == SplitTagging_Name}
    # assert SplitTagging_Name in AIDevName_to_Settings['SplitTagging']

    # Filtering_Name = AIDev_Args['Filtering_Name']
    # AIDevName_to_Settings['Filtering'] = {k: v for k, v in AIDevName_to_Settings['Filtering'].items() if k == Filtering_Name}
    # assert Filtering_Name in AIDevName_to_Settings['Filtering']


    # AIDataSettings = {
    #     'InputName_to_Settings': InputName_to_Settings,
    #     'TasksName_to_Settings': TasksName_to_Settings,
    #     'AIDevName_to_Settings': AIDevName_to_Settings,
    # }
    
    AIDataSettingInfo = {
        # 'AIDataSettings': AIDataSettings,
        # 'AIData_Job_Args': AIData_Job_Args,
        'OneAIDataName_to_OneAIDataArgs': OneAIDataName_to_OneAIDataArgs,
        'AIDataArgs_columns': AIDataArgs_columns, 
    }
    
    return AIDataSettingInfo
