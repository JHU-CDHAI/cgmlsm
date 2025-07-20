import os
import itertools
import datasets
import logging
import gc
import numpy as np
import pandas as pd
from datetime import datetime 
from tqdm import tqdm
from ..base import Base
from .casefnutils.hrf import get_HRFDirectory_from_HumanDirectory, get_HumanDirectoryArgs_ForOneCase, get_HumanDirectoryArgs_ForBatch
from .casefnutils.ro import parse_ROName, get_RONameToROInfo, get_RONameToROData_for_OneCaseExample
from .casefnutils.casefn import get_CaseFnNameToCaseFnInfo, get_CaseFnNameToCaseFnData_for_OneCaseExample

from datasets import disable_caching
disable_caching()

logger = logging.getLogger(__name__)



########## CaseFn Name Processing ##########
def get_HumanRecordRecfeatArgs_from_RONameToRONameInfo(ROName_to_RONameInfo):

    HumanRecordRecfeat_Args = {}

    for ROName, RONameInfo in ROName_to_RONameInfo.items():
        HumanName = RONameInfo['HumanName']
        if HumanName not in HumanRecordRecfeat_Args:
            HumanRecordRecfeat_Args[HumanName] = {}
            HumanRecordRecfeat_Args[HumanName][HumanName] = []

        RecordName = RONameInfo['RecordName']
        if RecordName not in HumanRecordRecfeat_Args[HumanName]:
            HumanRecordRecfeat_Args[HumanName][RecordName] = []

        if 'RecFeatName' in RONameInfo:
            RecFeatName = RONameInfo['RecFeatName']
            if RecFeatName not in HumanRecordRecfeat_Args[HumanName][RecordName]:
                HumanRecordRecfeat_Args[HumanName][RecordName].append(RecFeatName)
    return HumanRecordRecfeat_Args



def compare_dicts(dict1, dict2):
    """Returns a list of keys with differing values, or empty if equal."""
    diff_keys = []
    all_keys = set(dict1.keys()).union(dict2.keys())
    for key in all_keys:
        if dict1.get(key) != dict2.get(key):
            diff_keys.append(key)
    return diff_keys


def combine_Ckpd_to_CkpdObsConfig(CaseFnName_to_CaseFnInfo):
    combined_config = {}

    for case_fn_name, info in CaseFnName_to_CaseFnInfo.items():
        for ckpd, config in info.get('Ckpd_to_CkpdObsConfig', {}).items():
            if ckpd in combined_config:
                existing_config = combined_config[ckpd]
                if existing_config != config:
                    diff_keys = compare_dicts(existing_config, config)
                    raise ValueError(
                        f"[Conflict in '{ckpd}' from '{case_fn_name}'] "
                        f"Different config values for keys: {diff_keys}\n"
                        f"Existing: {existing_config}\nNew: {config}"
                    )
            else:
                combined_config[ckpd] = config

    return combined_config

def combine_ROName_list(CaseFnName_to_CaseFnInfo):
    combined_roinfo = {}

    for case_fn_name, info in CaseFnName_to_CaseFnInfo.items():
        for ro_name, ro_info in info.get('ROName_to_RONameInfo', {}).items():
            if ro_name in combined_roinfo:
                if combined_roinfo[ro_name] != ro_info:
                    raise ValueError(
                        f"[Conflict in ROName '{ro_name}' from '{case_fn_name}'] "
                        f"Inconsistent RONameInfo.\n"
                        f"Existing: {combined_roinfo[ro_name]}\n"
                        f"New: {ro_info}"
                    )
            else:
                combined_roinfo[ro_name] = ro_info


    ROName_list = [i for i in combined_roinfo]
    return ROName_list



def get_CaseFnTaskArgs_from_CaseFnNameList(CaseFnName_list, onecohort_record_base = None, SPACE = None):
    CaseFnName_to_CaseFnInfo = get_CaseFnNameToCaseFnInfo(CaseFnName_list, SPACE)
    ROName_list = combine_ROName_list(CaseFnName_to_CaseFnInfo)
    Ckpd_to_CkpdObsConfig = combine_Ckpd_to_CkpdObsConfig(CaseFnName_to_CaseFnInfo)

    ROName_to_RONameInfo = {
        ROName: parse_ROName(ROName) 
        for ROName in ROName_list
    }

    HumanRecordRecfeat_Args = get_HumanRecordRecfeatArgs_from_RONameToRONameInfo(ROName_to_RONameInfo)

    if onecohort_record_base is not None:
        ROName_to_ROInfo        = get_RONameToROInfo(ROName_list, onecohort_record_base, Ckpd_to_CkpdObsConfig)
    else:
        ROName_to_ROInfo = {}
    
    CaseFnTaskArgs = {
        'CF_list': CaseFnName_list,
        'CaseFnName_to_CaseFnInfo': CaseFnName_to_CaseFnInfo,
        'ROName_list': ROName_list,
        'Ckpd_to_CkpdObsConfig': Ckpd_to_CkpdObsConfig,
        
        'HumanRecordRecfeat_Args': HumanRecordRecfeat_Args,

        'ROName_to_RONameInfo': ROName_to_RONameInfo,
        'ROName_to_ROInfo': ROName_to_ROInfo,
    }
    return CaseFnTaskArgs


class Batch:
    def __init__(self, **kwargs):
        self.HRFDirectory = kwargs.get('HRFDirectory', {})
        self.RCKPD_to_Cache = kwargs.get('RCKPD_to_Cache', {})
        self.RO_to_Cache = kwargs.get('RO_to_Cache', {})
        self.CO_to_Cache = kwargs.get('CO_to_Cache', {})
        self.CF_to_Cache = kwargs.get('CF_to_Cache', {})
        
   
def get_ROCOCFData_for_OneCaseExample(case_example, 
                                      
                                      #################
                                      CF_list = None, 
                                      HRFDirectory = None, 
                                      CaseFnTaskArgs_to_Execute = None, # case.ObsNameList_to_Execute
                                      onecohort_record_base = None,
                                      #################

                                      caseset = None,     # <---- a collection of CF_list, ObsNameList_to_Execute, onecohort_record_base
                                      batch = None,       # <---- a collection of HRFDirectory, RO_to_Cache, RCKPD_to_Cache
                                      ):
    
    # caseset, a group of cases 
    # batch, within a caseset, a group of small number of cases
    # casefntask, above casests, a group of casesets to execute. # the definition of casefntask here is not clear.

    # ------------------- 1. [CaseSet Level] prepare the CF_list -------------------
    if CF_list is None:
        assert caseset is not None
        if caseset.CF_list_to_execute is not None:
            CF_list = caseset.CF_list_to_execute
        else:
            CF_list = caseset.CF_list


    # ------------------- 2. [CaseSet Level] CaseFnTaskArgs_to_Execute: at Be care about what to Observer  -------------------
    if CaseFnTaskArgs_to_Execute is None:
        assert caseset is not None
        CaseFnTaskArgs_to_Execute = caseset.CaseFnTaskArgs_to_Execute


    # -------------------- 3. HRDirectory -----------------------------------
    if HRFDirectory is None:

        # if casefntask.HRFDirectory is not None:
        #     # ------------------- 3.1 [Task Level] task.HRFDirectory is very expensive, it takes a lot of MEM -------------------
        #     # logger.info(f'HRFDirectory is available at Task Level')
        #     HRFDirectory = casefntask.HRFDirectory
        #     # print(HRFDirectory)
            
        if batch is not None:
            # ------------------- 3.2 [Batch Level]  -------------------
            # logger.info(f'HRFDirectory is available at Batch Level')
            HRFDirectory = batch.HRFDirectory
            assert len(HRFDirectory) > 0
            # print([i for i in HRFDirectory])
        
        else:
            # ------------------ 3.3 [OneCase Level] If HRFDirectory is not available, build it for One Case Level -----------
            # logger.info(f'HRFDirectory is not available, build it for One Case Level')
            HumanRecordRecfeat_Args = CaseFnTaskArgs_to_Execute['HumanRecordRecfeat_Args']
            if onecohort_record_base is None:
                assert caseset is not None
                onecohort_record_base = caseset.onecohort_record_base
            HumanDirectory_Args = get_HumanDirectoryArgs_ForOneCase(case_example, HumanRecordRecfeat_Args)
            HRFDirectory = get_HRFDirectory_from_HumanDirectory(onecohort_record_base, HumanDirectory_Args, HumanRecordRecfeat_Args)


    # logger.info(f'HRFDirectory {HRFDirectory}')

    # -------------------- 4. ROName List to Execute -----------------------------------
    # logger.info(task.RO_to_Cache)
    ROName_list = CaseFnTaskArgs_to_Execute['ROName_list']
    ROName_to_ROInfo = {ROName: CaseFnTaskArgs_to_Execute['ROName_to_ROInfo'][ROName] for ROName in ROName_list}
    
       
    if batch is not None:
        RO_to_Cache = batch.RO_to_Cache
        RCKPD_to_Cache = batch.RCKPD_to_Cache

    else:
        RO_to_Cache = {}
        RCKPD_to_Cache = {}


    ############################################################
    ROName_to_ROData = get_RONameToROData_for_OneCaseExample(case_example, 
                                                            ROName_to_ROInfo, 
                                                            HRFDirectory, 
                                                            RO_to_Cache, 
                                                            RCKPD_to_Cache,
                                                            caseset)
    # print(HRFDirectory)
    # print(ROName_to_ROInfo)
    # print(ROName_to_ROData)
    # raise Exception('stop here')

    # TODO: this is not correct, we need to get the CaseFnName_to_CaseFnInfo from the caseset
    CaseFnName_to_CaseFnInfo = CaseFnTaskArgs_to_Execute['CaseFnName_to_CaseFnInfo']
    CaseFnNameField_to_CaseFnData = get_CaseFnNameToCaseFnData_for_OneCaseExample(case_example, 
                                                                            CaseFnName_to_CaseFnInfo,
                                                                            ROName_to_ROInfo,
                                                                            ROName_to_ROData,
                                                                            # CO_to_Cache,
                                                                            caseset,
                                                                            )
    ############################################################
    return CaseFnNameField_to_CaseFnData


 
def get_ROCOCFData_for_CaseBatch(df_case_batch, 
                                 
                                 #################
                                 CF_list = None, 
                                 HRFDirectory = None, 
                                 CaseFnTaskArgs_to_Execute = None, # case.ObsNameList_to_Execute
                                 onecohort_record_base = None,
                                 #################
                            
                                 caseset = None,  # <---- a collection of CF_list, ObsNameList_to_Execute, onecohort_record_base
                                 # casefntask
                                 ):
    # caseset, a group of cases 
    # batch, within a caseset, a group of small number of cases
    # task, above casests, a group of casesets to execute. # the definition of task here is not clear.
    # ------------------- 1. [CaseSet Level] prepare the CF_list -------------------
    # logger.info(f'get_ROCOCFData_for_CaseBatch')
    # s1 = datetime.now()
    if CF_list is None:
        assert caseset is not None
        if caseset.CF_list_to_execute is not None:
            CF_list = caseset.CF_list_to_execute
        else:
            CF_list = caseset.CF_list


    # ------------------- 2. [CaseSet Level] ObsNameList_to_Execute: at Be care about what to Observer  -------------------
    if CaseFnTaskArgs_to_Execute is None:
        assert caseset is not None, f'caseset {caseset} is None'
        CaseFnTaskArgs_to_Execute = caseset.CaseFnTaskArgs_to_Execute
        # print(CaseFnTaskArgs_to_Execute)

    # e1 = datetime.now()
    # logger.info(f'cf and casefntask to execute: {e1-s1}')


    # df_case_batch = pd.DataFrame(case_examples)
    # for i, case_example in case_examples.iterrows():
    # s2 = datetime.now()
    # case_examples = {k: v for k, v in case_examples.items()}
    # df_case_batch = pd.DataFrame(dict(case_examples))
    # print(type(case_examples))
    # print(case_examples)
    # print(df_case_batch)
    # print(df_case_batch.shape)
    # e2 = datetime.now()
    # logger.info(f'case_examples to df_case_batch: {e2-s2}')



    # s3 = datetime.now()
    HumanRecordRecfeat_Args = CaseFnTaskArgs_to_Execute['HumanRecordRecfeat_Args']
    onecohort_record_base = caseset.onecohort_record_base
    HumanDirectory_Args = get_HumanDirectoryArgs_ForBatch(df_case_batch, HumanRecordRecfeat_Args)
    # e3 = datetime.now()
    # logger.info(f'get_HumanDirectoryArgs_ForBatch: {e3-s3}')



    # s4 = datetime.now()
    HRFDirectory = get_HRFDirectory_from_HumanDirectory(onecohort_record_base, HumanDirectory_Args, HumanRecordRecfeat_Args)
    # e4 = datetime.now()
    # logger.info(f'get_HRFDirectory_from_HumanDirectory: {e4-s4}')

    # print('HumanDirectory_Args:', HumanDirectory_Args)
    # print('HumanRecordRecfeat_Args:', HumanRecordRecfeat_Args)
    # print([i for i in HRFDirectory])
    # print(HRFDirectory)
    # raise Exception('stop here')


    # s5 = datetime.now()
    batch_dict = {'HRFDirectory': HRFDirectory, 'RCKPD_to_Cache': {}, 'RO_to_Cache': {}, 'CO_to_Cache': {}, 'CF_to_Cache': {}}
    batch = Batch(**batch_dict)
    # e5 = datetime.now()
    # logger.info(f'batch: {e5-s5}')
    # print([i for i in batch.HRFDirectory])
    # print([i for i in batch.HRFDirectory])
    # print(batch.HRFDirectory)
    

    # s6 = datetime.now()
    for i, case_example in df_case_batch.iterrows():
        CF_Data_Final = get_ROCOCFData_for_OneCaseExample(case_example, caseset = caseset, batch = batch)
        
        # logger.info(f'CF_Data_Final {CF_Data_Final}')
        for CF_seqtype, SeqValues in CF_Data_Final.items():
            # print(CF_seqtype, SeqValues)
            # case_examples.at[i, CF] = SeqValues
            if CF_seqtype not in df_case_batch.columns:
                df_case_batch[CF_seqtype] = None
            df_case_batch.at[i, CF_seqtype] = SeqValues
    # e6 = datetime.now()
    # logger.info(f'get_ROCOCFData_for_OneCaseExample: {e6-s6}')

    CFtype_list = [i for i in CF_Data_Final]

    # logger.info(f'df_case_batch')
    # logger.info(df_case_batch)
    # logger.info(sorted(list(df_case_batch.columns)))
    df_case_batch_cf = df_case_batch[CFtype_list]# .to_dict(orient='list')
    return df_case_batch_cf


class CaseSetIO(Base):
    def get_caseset_path(self):
        caseset_path = os.path.join(
            self.cohort_casebase_folder, self.CaseCollectionName, self.CaseSetName
        )
        return caseset_path

    # check caseset_path
    def save_CFs_to_disk(self):
        # CaseSetNameCF_to_Data = {}
        # CaseSetName = caseset.CaseSetName
        caseset_path = self.caseset_path
        if not os.path.exists(caseset_path):
            os.makedirs(caseset_path)

        ds_case = self.ds_case
        columns = ds_case.column_names
        CF_list_to_execute = self.CF_list_to_execute
        for CF in CF_list_to_execute:
            caseset_cf_path = os.path.join(caseset_path, CF)
            if os.path.exists(caseset_cf_path):
                # maybe add some logs here.
                logger.info(f'Already in disk: {caseset_cf_path}')
                
            columns_cf = [i for i in columns if CF+'-' in i]
            Data_CF = ds_case.select_columns(columns_cf)
            Data_CF.save_to_disk(caseset_cf_path)
            logger.info(f'Save to disk: {caseset_cf_path}')


    # check caseset_path
    def load_CFs_from_disk(self, CF_list):
        # CF_list = self.CF_list
        case_id_columns = self.case_id_columns
        ds_case = self.ds_case.select_columns(case_id_columns)  

        # assert os.path.exists(caseset_path)
        Data_CF_list = [ds_case]
        CF_list_to_execute = []
        CF_list_available = []
        caseset_path = self.caseset_path
        for CF in CF_list:
            caseset_cf_path = os.path.join(caseset_path, CF)
            if os.path.exists(caseset_cf_path):
                Data_CF = datasets.Dataset.load_from_disk(caseset_cf_path)
                # logger.info(f'Load from disk: {Data_CF} for CF: {CF}')
                columns = [i for i in Data_CF.column_names if CF+'-' in i] 
                Data_CF = Data_CF.select_columns(columns)
                assert len(Data_CF) == len(ds_case)
                Data_CF_list.append(Data_CF)
                CF_list_available.append(CF)
            else:
                CF_list_to_execute.append(CF)

        if len(Data_CF_list) > 1:
            
            s = datetime.now()
            # for i in Data_CF_list: logger.info(i)
            # logger.info(f'concatenate_datasets -- CFs: {CF_list_available}')
            ds_case = datasets.concatenate_datasets(Data_CF_list, axis = 1)
            
            e = datetime.now()
            logger.info(f'concatenate_datasets -- time: {e - s} for CFs: {CF_list_available}')
            
        else:
            ds_case = ds_case 

        if len(CF_list_available) == len(CF_list):
            need_to_execute_obs = False
        else:
            need_to_execute_obs = True

        load_info = {
            'ds_case': ds_case,
            'CF_list_available': CF_list_available,
            'CF_list_to_execute': CF_list_to_execute,
            'need_to_execute_obs': need_to_execute_obs
        }
        return load_info # ds_case, CF_list_available, CF_list_to_execute, need_to_execute_obs
    

    def remove_ds_case(self):
        # Make sure the attribute exists before trying to delete it
        if hasattr(self, 'ds_case'):
            del self.ds_case
        gc.collect()


class CaseSet(CaseSetIO):
    
    def __init__(self, 
                 onecohort_record_base,
                 Trigger_Args, 
                 df_case,
                 # Case_Args_Settings = None,
                 caseset_metainfo = None, 
                 Case_Proc_Config = None, 
                 SPACE = None 
                 ):
        
        # 1. onecohort_record_base 
        self.SPACE = SPACE
        self.onecohort_record_base = onecohort_record_base

        # 2. Trigger_Args
        self.caseset_metainfo = caseset_metainfo
        # Trigger_Args = caseset_metainfo['Trigger_Args']
        self.Trigger_Args = Trigger_Args
        self.TriggerName = Trigger_Args['Trigger']
        self.case_id_columns = Trigger_Args['case_id_columns']
        self.ObsDTName = Trigger_Args['ObsDT']
        self.HumanID_list = Trigger_Args['HumanID_list']


        # 2.2 Folder and Path
        SPACE = self.onecohort_record_base.SPACE
        if caseset_metainfo is not None:
            self.cohort_casebase_folder = caseset_metainfo['cohort_casebase_folder'].replace('$DATA_CASE$', SPACE.get('DATA_CASE', '--'))
            self.CaseCollectionName = caseset_metainfo['CaseCollectionName']
            self.CaseSetName = self.get_CaseSetName(df_case, Trigger_Args, caseset_metainfo)
            self.caseset_path = self.get_caseset_path()

        # 3. Case_Args_Settings
        # self.Case_Args_Settings = Case_Args_Settings
        self.Case_Proc_Config = Case_Proc_Config

        
        # 4. develop df_case and ds_case.
        # we have df_case_ids, df_case, ds_case

        #################
        # TODO: in the future, maybe just ds_case, could be dataframe or huggingface-dataset
        ################
        self.df_case = df_case
        if Case_Proc_Config['via_method'] == 'ds':
            self.ds_case = datasets.Dataset.from_pandas(df_case)
        else:
            self.ds_case = df_case

        # CF related features
        # self.CF_to_CFArgs   = Case_Args_Settings['CF_to_CFArgs']
        # self.TagCF_to_TagCFArgs = Case_Args_Settings['TagCF_to_TagCFArgs']

        ######################
        self.CF_list = None 
        self.TagCF_list = None
        self.CF_list_to_execute = None
        self.TagCF_list_to_execute = None
        self.ObsNameList_to_Execute = None 
        self.CaseFnTaskArgs = None 
        ######################


    # ---------- CaseFnTask: Building ----------
    def build_CaseFnTaskArgs(self, 
                             CF_list, #### <---- CF list 
                             Case_Proc_Config, 
                             onecohort_record_base = None,
                             SPACE= None,
                             set_attribute = True, 
                             ):
        
        if onecohort_record_base is None:
            onecohort_record_base = self.onecohort_record_base
        if SPACE is None:
            SPACE = self.onecohort_record_base.SPACE

        CaseFnTaskArgs = get_CaseFnTaskArgs_from_CaseFnNameList(CF_list, onecohort_record_base = onecohort_record_base, SPACE = SPACE)
        self.HumanRecordRecfeat_Args = CaseFnTaskArgs['HumanRecordRecfeat_Args']
        self.CaseFnName_to_CaseFnInfo = CaseFnTaskArgs['CaseFnName_to_CaseFnInfo']
        
        if set_attribute == True:
            self.CaseFnTaskArgs = CaseFnTaskArgs
            self.Case_Proc_Config = Case_Proc_Config
        
        return CaseFnTaskArgs, Case_Proc_Config
    

    def get_CaseSetName(self, df_case, Trigger_Args, caseset_metainfo):
        TriggerName = Trigger_Args['Trigger'] 
        GroupName = caseset_metainfo['GroupName']   
        max_case_num = caseset_metainfo['max_case_num']
        idx = caseset_metainfo['idx']

        caseset_chunk_size = caseset_metainfo['caseset_chunk_size']
        if max_case_num is not None: max_case_num = 'All'

        digital_num = len(str(len(df_case)))
        idx_start = caseset_chunk_size * idx
        idx_start_str = str(idx_start).zfill(digital_num)
        idx_end = idx_start + len(df_case)
        idx_end_str = str(idx_end).zfill(digital_num)
        CaseSetName = 'T.' + TriggerName + '--' + 'G.' + GroupName + f'--chunk.{idx_start_str}-{idx_end_str}'
        return CaseSetName  
    

    def execute_casefn_task(self, 
                            CaseFnTaskArgs = None, 
                            Case_Proc_Config = None
                           ):
        
        s = datetime.now()
        logger.info(f'----------- ** <{self.CaseSetName}> ** -----------')  
        logger.info(f'df_case: {self.df_case.shape}')

        # ----------------- CaseFnTaskArgs -----------------
        if CaseFnTaskArgs is None:
            CaseFnTaskArgs = self.CaseFnTaskArgs
        if Case_Proc_Config is None:
            Case_Proc_Config = self.Case_Proc_Config
        save_data =  Case_Proc_Config['save_data']
        load_data =  Case_Proc_Config['load_data']


        # ----------------- CaseFnTaskArgs -----------------
        # CaseFnTaskArgs = casefn_task.CaseFnTaskArgs
        CF_list = CaseFnTaskArgs['CF_list']
        logger.info(f'task to execute for CF_list: {CF_list}')
        self.CaseFnTaskArgs = CaseFnTaskArgs


        CF_list = CaseFnTaskArgs['CF_list']
        self.CF_list = CF_list

        # ----------------- CF_list -----------------
        CF_list_all = CF_list # + [TagCF_to_TagCFArgs[TagCFName]['CF'] for TagCFName in TagCF_list_to_execute]
        CF_list_all = list(set(CF_list_all))  
        if len(CF_list_all) > 0 and load_data == True: 
            # logger.info(f'Load CFs from disk: {CF_list_all}')
            # ds_case, CF_list_available, CF_list_to_execute, need_to_execute_obs 
            load_info = self.load_CFs_from_disk(CF_list_all)
            ds_case = load_info['ds_case']
            self.ds_case = ds_case 
            # logger.info(f'ds_case {ds_case}')
            # self.CF_list_available = load_info['CF_list_available']
            CF_list_to_execute = load_info['CF_list_to_execute']
        else:
            CF_list_to_execute = CF_list_all
        logger.info(f'CF_list_to_execute:\t {CF_list_to_execute}')

        # logger.info(f'2. df_case.columns: {selfdf_case.columns}')


        # ----------------- Execute CFs -----------------
        self.CF_list_to_execute = CF_list_to_execute
        # self.TagCF_list_to_execute = TagCF_list_to_execute
        # self.ObsNameList_to_Execute = get_ROCOGammePhiInfo_from_CFList(CF_list_to_execute, self.CF_to_CFArgs)


        CaseFnTaskArgs_to_Execute, Case_Proc_Config = self.build_CaseFnTaskArgs(CF_list_to_execute, Case_Proc_Config, set_attribute = False)
        self.CaseFnTaskArgs_to_Execute = CaseFnTaskArgs_to_Execute
        # print(self.CaseFnTaskArgs_to_Execute)
        # print('CF_list_to_execute:', CF_list_to_execute)
        # print('Case_Proc_Config:', Case_Proc_Config)


        # logger.info(f'3. df_case.columns: {df_case.columns}')
        # Try to execute the CFs
        if len(CF_list_to_execute) > 0:
            CF_list_to_execute = self.CF_list_to_execute
            # ObsNameList_to_Execute = caseset.ObsNameList_to_Execute
            logger.info(f'... Executing: CF_list_to_execute {CF_list_to_execute}')
            ds_case = self.execute_casefn_task_for_CFs()
            self.ds_case = ds_case


        if save_data == True and len(CF_list_to_execute) > 0: 
            self.save_CFs_to_disk()

        # logger.info(f'6. df_case.columns: {df_case.columns}')

        e = datetime.now()
        du = e - s
        logger.info(f'----------- ** Finish CaseFnTask for CaseSet <{self.CaseSetName}>-- time: {du} ** -----------\n')


    # ----------------------------- CaseFnTask Part ----------------------------
    def execute_casefn_task_for_CFs(self):
        # ------------------------------------
        caseset = self
        ds_case = caseset.ds_case
        case_id_columns = caseset.case_id_columns
        # ------------------------------------

        Case_Proc_Config = caseset.Case_Proc_Config
        via_method = Case_Proc_Config['via_method']

        if via_method == 'ds': 
            s = datetime.now()
            assert all([i in ds_case.column_names for i in case_id_columns])
            assert type(ds_case) == datasets.Dataset    

            batch_size = Case_Proc_Config['batch_size']
            # logger.info(f'batch_size: {batch_size} and in via_method: {via_method}')
            
            # ------------------------------------
            # ds_case = ds_case.map(lambda case_examples: get_ROCOCFData_for_CaseBatch(case_examples, caseset = caseset),
            #                         batched=True,
            #                         batch_size=batch_size,  # Process 1000 cases at a time
            #                         load_from_cache_file=False)
            # ------------------------------------
            

            # ------------------------------------ # TODO: consider the smart choice of hf dataset or pandas dataframe
            df_case = ds_case.to_pandas()
            total_rows = len(df_case)
            batches = [df_case.iloc[i:min(i + batch_size, total_rows)] for i in range(0, total_rows, batch_size)]
            assert sum(len(i) for i in batches) == len(df_case)
            results = []

            for idx, df_case_batch in tqdm(enumerate(batches)):
                if len(df_case_batch) == 0: continue
                df_case_batch = df_case_batch.reset_index(drop = True)
                df_case_batch_cf = get_ROCOCFData_for_CaseBatch(df_case_batch, caseset = caseset)
                results.append(df_case_batch_cf)

            df_case_cf = pd.concat(results).reset_index(drop = True)
            ds_case = pd.concat([df_case, df_case_cf], axis=1)
            ds_case = datasets.Dataset.from_pandas(ds_case)
            # ------------------------------------

            e = datetime.now()
            logger.info(f'execute_casefn_task_via_ds {self.CaseSetName} -- time: {e - s}')

        elif via_method == 'df':
            s = datetime.now()
            assert all([i in ds_case.columns for i in case_id_columns])
            # if type(ds_case) != pd.DataFrame: ds_case = ds_case.to_pandas()

            logger.info(f'batch_size: None and in via_method: {via_method}')
            batch_size = Case_Proc_Config['batch_size']
            
            df_case_cf = pd.DataFrame(ds_case.apply(lambda case_example: get_ROCOCFData_for_OneCaseExample(case_example, caseset = caseset),
                                                axis = 1).to_list())
            ds_case = pd.concat([ds_case, df_case_cf], axis = 1)
            e = datetime.now()
            logger.info(f'execute_casefn_task_via_df {self.CaseSetName} -- time: {e - s}')

        else:
            raise ValueError(f'via_method: {via_method} is not supported.')
        
        return ds_case
        

    def __repr__(self):
        return f"<{self.CaseSetName}>"
    