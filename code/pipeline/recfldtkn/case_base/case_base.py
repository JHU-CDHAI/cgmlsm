# === Standard Library ===
import os
import sys
import json
import copy
import math
import random
import pickle
import logging
import multiprocessing as mp
from multiprocessing import shared_memory
import datasets
import gc
import resource

# === External Libraries ===
import numpy as np
import pandas as pd
import torch
from pympler import asizeof, tracker
from datasets import DatasetInfo

# === Project Base ===
from ..base import Base, apply_condition, apply_multiple_conditions

# === Case Set and Utilities ===
from .caseset import CaseSet
from .casefnutils.casefn import Case_Fn

# === Random Seed for Reproducibility ===
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



logger = logging.getLogger(__name__)
CASESET_TRIGGER_PATH = 'fn/fn_case/trigger'




## ---------------- case set trigger task ----------------
class CaseSet_Trigger_Fn(Base):
    def __init__(self, TriggerName, SPACE):
        self.SPACE = SPACE
        pypath = os.path.join(self.SPACE['CODE_FN'], CASESET_TRIGGER_PATH, TriggerName + '.py')
        self.pypath = pypath
        self.load_pypath()

    def load_pypath(self):
        module = self.load_module_variables(self.pypath)
        self.Trigger_Args = module.Trigger_Args
        self.get_CaseTrigger_from_RecordBase = module.get_CaseTrigger_from_RecordBase


## ---------------- case splitting -----------------
def split_dataframe(df, Case_Proc_Config):
    n_cpus = Case_Proc_Config['n_cpus']
    max_chunk_size = Case_Proc_Config['caseset_chunk_size']
    total_len = len(df)

    # Adjust chunk size to ensure at least n_cpus chunks
    chunk_size = min(max_chunk_size, math.ceil(total_len / n_cpus))

    # Generate start indices for each chunk
    indices = np.arange(0, total_len, chunk_size)

    # Split the DataFrame using numpy indexing
    df_small_list = [df.iloc[i : i + chunk_size] for i in indices]


    Case_Proc_Config['max_chunk_size_updated'] = chunk_size

    return df_small_list, Case_Proc_Config




def get_CaseCollectionName_Info(TriggerCaseBase_Args, Case_Proc_Config, suffix = ''):

    max_case_num = Case_Proc_Config.get('max_trigger_case_num', None)
    
    TriggerType = 'TriggerAll' if max_case_num is None else f'Trigger{max_case_num}'
    caseset_chunk_size = Case_Proc_Config['caseset_chunk_size']
    TriggerType = f'{TriggerType}.cs{caseset_chunk_size}'

    # --------- Trigger's CaseCollectionArgs ----------
    CaseCollectionArgs = TriggerCaseBase_Args['Trigger']
    FilterName = CaseCollectionArgs.get('Filter', None)
    GroupName = CaseCollectionArgs.get('Group', None)
    
    Name_list = [TriggerType, FilterName, GroupName] 
    Name_list = [i for i in Name_list if i is not None] 

    CaseCollectionName = '-'.join(Name_list) + suffix
    CaseCollectionName_list = [CaseCollectionName]
    CaseCollectionName_to_CaseCollectionArgs = {CaseCollectionName: TriggerCaseBase_Args['Trigger']}

    FilterCaseCollection_list = [i for i in TriggerCaseBase_Args if 'FilterCaseSet' in i]
    for FilterCaseCollectionName in FilterCaseCollection_list:
        CaseCollectionArgs = TriggerCaseBase_Args[FilterCaseCollectionName]
        FilterName = CaseCollectionArgs['Filter']
        CaseCollectionName = CaseCollectionName + f'-{FilterName}'
        CaseCollectionName_list.append(CaseCollectionName)
        CaseCollectionName_to_CaseCollectionArgs[CaseCollectionName] = CaseCollectionArgs


    CaseCollectionName_Info = {
        'CaseCollectionName_list': CaseCollectionName_list,
        'CaseCollectionName_to_CaseCollectionArgs': CaseCollectionName_to_CaseCollectionArgs,
    }
    
    return CaseCollectionName_Info




## ---------------- tag rec task ----------------
def get_dfTagRec_from_TagRecArgs(TagRecArgs, onecohort_record_base = None):
    HumanName = TagRecArgs['HumanName']
    RecordName = TagRecArgs['RecordName']
    columns_to_group = TagRecArgs['columns_to_group']
    columns_to_tag = TagRecArgs['columns_to_tag']
    post_op = TagRecArgs['post_op']
    record = onecohort_record_base.Name_to_HRF[(HumanName, RecordName)]

    columns = columns_to_group + columns_to_tag
    try:
        df_RecAttr = record.ds_RecAttr.select_columns(columns).to_pandas()
        df_rectag = df_RecAttr.reset_index(drop = True)
    except:
        df_RecAttr = record.df_RecAttr
        df_rectag = df_RecAttr[columns].reset_index(drop = True)

    post_op = TagRecArgs['post_op']
    if post_op == 'max':
        df_rectag = df_rectag.groupby(columns_to_group).max().reset_index()
    elif post_op is None:
        df_rectag = df_rectag
    elif post_op == 'unique-str-join':

        # len1 = len(df_rectag)
        df_rectag_index = df_rectag[columns_to_group].drop_duplicates() 
        # len2 = len(df_rectag_index)
        # assert len1 == len2, f'len1: {len1}, len2: {len2}'
        for column in columns_to_tag:
            # df_rectag[column] = df_rectag[column].astype(str)
            # df_rectag[column] = df_rectag[column].fillna("").astype(str)

            # df_rectag[columns_to_group] = df_rectag[columns_to_group].applymap(lambda x: str(x).strip() if pd.notna(x) else x)
            df_rectag[column] = df_rectag[column].fillna("").astype(str)

            df_rectag_column = df_rectag.groupby(columns_to_group)[column].apply(lambda x: '|'.join(set(x))).reset_index()
            df_rectag_column = df_rectag_column.rename(columns = {column: column + ':unique_join'})
            # li.append(df_rectag_columns)
            # if len(df_rectag_index) != len(df_rectag_column):
            #     # print(df_rectag.columns)
            #     print(df_rectag_index.columns)
            #     print(df_rectag_column.columns)
            
            assert len(df_rectag_index) == len(df_rectag_column), f'len(df_rectag_index): {len(df_rectag_index)}, len(df_rectag_column): {len(df_rectag_column)}'
            df_rectag_index = pd.merge(df_rectag_index, df_rectag_column, how = 'left')
            # assert len(df_rectag_index) == len(df_rectag_column), f'len(df_rectag_index): {len(df_rectag_index)}, len(df_rectag_column): {len(df_rectag_column)}'


        df_rectag = df_rectag_index

    elif post_op == 'count': 
        # df_rectag = df_rectag.groupby(columns_to_group).count().reset_index()
        df_rectag = df_rectag[columns_to_group].value_counts().reset_index()
       
        df_rectag = df_rectag.rename(columns = {'count': f'{HumanName}-{RecordName}:count'})
        # print(df_rectag)
    else:
        raise ValueError(f'Unsupported post_op: {post_op}')
    return df_rectag


def execute_tagrectask(df_case,
                    onecohort_record_base, 
                    TagRecName,
                    cohort_casebase_folder,
                    TagRec_to_TagRecArgs, 
                    Case_Proc_Config
                    ):
    # for TagRecName in TagRec_list:
    task_data_file = os.path.join(cohort_casebase_folder, 'TagRec', f'{TagRecName}.p')
    
    redo_tagrec = Case_Proc_Config.get('redo_tagrec', False)
    if Case_Proc_Config['load_data'] == True and os.path.exists(task_data_file) and redo_tagrec == False:
        df_rectag = pd.read_pickle(task_data_file)
        # assert len(df_case) == len(df_rectag)
        df_case = pd.merge(df_case, df_rectag, how = 'left')
        logger.info(f'Load df_rectag of size {df_rectag.shape} from {task_data_file}')
    
    else:
        TagRecArgs = TagRec_to_TagRecArgs[TagRecName]
        df_rectag = get_dfTagRec_from_TagRecArgs(TagRecArgs, onecohort_record_base)
        before = len(df_case)
        df_case = pd.merge(df_case, df_rectag, how = 'left')
        after = len(df_case)
        assert before == after, f'before: {before}, after: {after}, TagRecName: {TagRecName}, df_rectag: {df_rectag.shape}'

        if Case_Proc_Config['save_data'] == True:
            os.makedirs(os.path.dirname(task_data_file), exist_ok = True)
            df_rectag.to_pickle(task_data_file)
            logger.info(f'Save df_rectag of size {df_rectag.shape} to {task_data_file}')
    return df_case 


## ---------------- cf task ----------------
def worker_init(onecohort_shm_name, onecohort_size):
    global casefn_task, onecohort_record_base

    # casefn_task_shm = shared_memory.SharedMemory(name=casefn_task_shm_name)
    onecohort_shm = shared_memory.SharedMemory(name=onecohort_shm_name)
    # casefn_task = pickle.loads(casefn_task_shm.buf[:casefn_task_size])
    onecohort_record_base = pickle.loads(onecohort_shm.buf[:onecohort_size])


def analyze_object_memory(obj, name="object"):
    logging.info(f"Memory analysis for {name}:")
    total_size = asizeof.asizeof(obj)
    logging.info(f"Total size: {total_size:,} bytes")
    
    if hasattr(obj, '__dict__'):
        for attr, value in obj.__dict__.items():
            size = asizeof.asizeof(value)
            logging.info(f"  {attr}: {size:,} bytes ({size/total_size:.2%} of total)")
    else:
        logging.info("  Object doesn't have a __dict__ attribute.")

    if isinstance(obj, (list, tuple, set)):
        logging.info(f"  Number of items: {len(obj)}")
        if len(obj) > 0:
            logging.info(f"  Average item size: {asizeof.asizeof(obj) / len(obj):,.2f} bytes")
    elif isinstance(obj, dict):
        logging.info(f"  Number of items: {len(obj)}")
        if len(obj) > 0:
            avg_key_size = sum(asizeof.asizeof(k) for k in obj.keys()) / len(obj)
            avg_value_size = sum(asizeof.asizeof(v) for v in obj.values()) / len(obj)
            logging.info(f"  Average key size: {avg_key_size:,.2f} bytes")
            logging.info(f"  Average value size: {avg_value_size:,.2f} bytes")



def log_object_size(obj, name):
    size_bytes = asizeof.asizeof(obj)
    size_gb = size_bytes / (1024 ** 3)
    logger.info(f"[MEMORY] {name}: {size_bytes} bytes ({size_gb:.2f} GB)")


def caseset_task_worker(caseset):
    try:
        # Log initial memory state
        logger.info("==== MEMORY USAGE REPORT ====")
        log_object_size(caseset, "caseset before processing")
        log_object_size(onecohort_record_base, "onecohort_record_base")
        # logger.info("==== END MEMORY REPORT ====")

        # Process the case set
        caseset.onecohort_record_base = onecohort_record_base
        caseset.execute_casefn_task()
        
        # Clean up the large object
        del caseset.onecohort_record_base
        
        # Explicit garbage collection
        gc.collect()
        
        # Log final memory state
        # logger.info("==== MEMORY USAGE REPORT ====")
        log_object_size(caseset, "caseset after processing")
        logger.info("==== END MEMORY REPORT ====")
        return caseset
    
    except Exception as e:
        logger.error(f"Error in worker: {e}")
        raise ValueError(f"Error in worker: {e}")




class CaseFn_Task(Base):

    def __init__(self, 
                 CaseSetName_to_caseset,
                 onecohort_record_base,
                 CF_list, # <---- this is based on a TriggerGrpFlt.
                 Case_Proc_Config,
                 SPACE = None, 
                 ):
        
        self.CaseSetName_to_caseset = CaseSetName_to_caseset
        self.onecohort_record_base = onecohort_record_base # OneCohort_Record_Base_NeatVersion()
        self.CF_list = CF_list
        self.Case_Proc_Config = Case_Proc_Config
        self.SPACE = SPACE

        for CaseSetName, caseset in self.CaseSetName_to_caseset.items():
            CaseFnTaskArgs, Case_Proc_Config = caseset.build_CaseFnTaskArgs(CF_list, Case_Proc_Config) # will be updated with CaseFnTaskArgs
            self.CaseSetName_to_caseset[CaseSetName] = caseset
            self.CaseFnTaskArgs = CaseFnTaskArgs
    

    def execute_casefn_tasks(self, 
                        CaseSetName_to_caseset = None, # <----- before enter this function, CaseSetName_to_caseset is already built with the task. 
                        CaseFnTaskArgs = None,
                        ):
        # task = self

        Case_Proc_Config = self.Case_Proc_Config 
        # use_task_cache = Case_Proc_Config['use_task_cache']
        # reload_after_process = Case_Proc_Config.get('reload_after_process', False)
        # save_data = Case_Proc_Config['save_data']   

        if CaseSetName_to_caseset is None:
            CaseSetName_to_caseset = self.CaseSetName_to_caseset
        if CaseFnTaskArgs is None:
            CaseFnTaskArgs = self.CaseFnTaskArgs
        
        for CaseSetName, caseset in CaseSetName_to_caseset.items():

            #########
            del caseset.onecohort_record_base
            # caseset is already updated with CaseFnTaskArgs
            onecohort_record_base = self.onecohort_record_base
            logger.info("==== MEMORY USAGE REPORT ====")

            log_object_size(caseset, "caseset")
            log_object_size(onecohort_record_base, "onecohort_record_base")
            # log_object_size(casefn_task, "casefn_task")  # Uncomment if needed
            logger.info("==== END MEMORY REPORT ====")
            caseset.onecohort_record_base = onecohort_record_base
            
            caseset.execute_casefn_task()
            #########
            
            CaseSetName_to_caseset[CaseSetName] = caseset

        return CaseSetName_to_caseset



    # convert this part to the multi-processing part.
    def execute_casefn_tasks_with_multiple_process(self, 
                                               CaseSetName_to_caseset = None, 
                                               CaseFnTaskArgs = None,
                                               ):

        if CaseSetName_to_caseset is None:
            CaseSetName_to_caseset = self.CaseSetName_to_caseset
        if CaseFnTaskArgs is None:
            CaseFnTaskArgs = self.CaseFnTaskArgs
        

        # Set up a pool of processes
        # obstask = self
        n_cpus = self.Case_Proc_Config['n_cpus']
        # Split the data into chunks for each process
        CaseSetName_list = list(CaseSetName_to_caseset.keys())
        round_num = len(CaseSetName_list) // n_cpus + 2
        onecohort_record_base = self.onecohort_record_base


        # Serialize task and onecohort_record_base
        # del obstask.onecohort_record_base
        # del obstask.CaseSetName_to_caseset
        # obstask_bytes = pickle.dumps(obstask)
        onecohort_record_base_bytes = pickle.dumps(onecohort_record_base)
        logger.info(f"Size of serialized onecohort_record_base: {len(onecohort_record_base_bytes):,} bytes")
    


        # logger.info(f"sys.getsizeof(obstask_bytes):               {sys.getsizeof(obstask_bytes)}")
        # logger.info(f"sys.getsizeof(onecohort_record_base_bytes): {sys.getsizeof(onecohort_record_base_bytes)}")


        # Create shared memory for task and onecohort_record_base
        # obstask_shm = shared_memory.SharedMemory(create=True, size=len(obstask_bytes))
        onecohort_shm = shared_memory.SharedMemory(create=True, size=len(onecohort_record_base_bytes))



        try:
            # Copy data to shared memory
            # obstask_shm.buf[:len(obstask_bytes)] = obstask_bytes
            onecohort_shm.buf[:len(onecohort_record_base_bytes)] = onecohort_record_base_bytes


            for i in range(round_num):
                
                start = i * n_cpus
                end = (i + 1) * n_cpus
                if start >= len(CaseSetName_list): break

                # logger.info(f"======= Processing chunk {i + 1} of {round_num} =======")
                logger.info(f"======= Processing chunk {i + 1} of {round_num} ({end-start} case sets) =======")
                CaseSetName_list_chunk = CaseSetName_list[start:end]
                CaseSetName_to_caseset_chunk = {CaseSetName: CaseSetName_to_caseset[CaseSetName] for CaseSetName in CaseSetName_list_chunk}
                # task.execute_obstasks_with_multiple_process_chunk(CaseSetName_to_caseset_chunk, ObsTaskArgs)

                with mp.Pool(n_cpus, 
                            initializer=worker_init, 
                            initargs=(onecohort_shm.name, 
                                    len(onecohort_record_base_bytes))
                            ) as pool:
                    
                    
                    jobs_list = []
                    for caseset in CaseSetName_to_caseset_chunk.values():
                        # ADDED: Check if attribute exists before deleting
                        if hasattr(caseset, 'onecohort_record_base'):
                            del caseset.onecohort_record_base
                            
                        # del caseset.onecohort_record_base
                        # caseset.onecohort_record_base = onecohort_record_base
                        jobs_list.append((caseset,))


                    # jobs_list = [(caseset,) for caseset in CaseSetName_to_caseset_chunk.values()]

                    try:
                        results = pool.starmap(caseset_task_worker, jobs_list)

                        # Update results
                        for case_set_name, updated_caseset in zip(CaseSetName_to_caseset_chunk.keys(), results):
                            updated_caseset.onecohort_record_base = onecohort_record_base
                            if self.Case_Proc_Config['save_memory'] == True:
                                updated_caseset.remove_ds_case()
                            CaseSetName_to_caseset[case_set_name] = updated_caseset


                    except Exception as e:
                        print(f"Error occurred during multiprocessing: {e}")
                        pool.terminate()
                        raise ValueError(f"Error occurred during multiprocessing: {e}")
                    finally:
                        pool.close()
                        pool.join()
                    
                # ADDED: Explicit cleanup between chunks
                del CaseSetName_to_caseset_chunk
                del jobs_list
                del results
                gc.collect()

                # ADDED: Memory tracking after each chunk
                logger.info(f"Memory after chunk {i+1}: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024:.2f} MB")
            

        except Exception as e:
            logger.error(f"Error occurred during multiprocessing: {e}")
            raise ValueError(f"Error occurred during multiprocessing: {e}")
        
        finally:
            # ADDED: Always clean up shared memory and serialized data
            onecohort_shm.close()
            onecohort_shm.unlink()
            del onecohort_record_base_bytes
            gc.collect()

        return CaseSetName_to_caseset
    


class OneCohortTrigger_CaseBase(Base):
    
    def __init__(self, 
                 onecohort_record_base, 
                 TriggerCaseBase_Args,
                 dataset_init = None, 
                 dataset_init_name = None,
                 Case_Proc_Config = None, 
                 Case_Args_Settings = None, 
                 SPACE = None, 
                 ):
        # onecohort_record_base 
        self.onecohort_record_base = onecohort_record_base
        self.CohortName = onecohort_record_base.CohortName
        
        # trigger case abse 
        self.TriggerCaseBase_Args = TriggerCaseBase_Args
        self.TriggerName = TriggerCaseBase_Args['Trigger']['TriggerName']
        
        #  Case Proc Config
        if Case_Proc_Config is None: Case_Proc_Config = {}
        self.Case_Proc_Config = Case_Proc_Config
        if 'save_memory' not in Case_Proc_Config:
            Case_Proc_Config['save_memory'] = False


        # Folder Information.
        self.cohort_casebase_folder = os.path.join(SPACE.get('DATA_CASE', ''), self.CohortName, self.TriggerName)


        # if dataset_init is None:
        #     self.cohort_casebase_folder = os.path.join(SPACE.get('DATA_CASE', ''), self.CohortName, self.TriggerName)
        # else:
        #     self.cohort_casebase_folder = os.path.join(SPACE.get('DATA_CASE', ''), self.CohortName, self.TriggerName + '_' + dataset_init_name + '_' + str(len(dataset_init)))


        suffix = ''
        if dataset_init is not None:
            suffix = '_' + dataset_init_name + '_' + str(len(dataset_init))

        CaseCollectionName_Info = get_CaseCollectionName_Info(TriggerCaseBase_Args, Case_Proc_Config, suffix)
        self.CaseCollectionName_list = CaseCollectionName_Info['CaseCollectionName_list']
        self.CaseCollectionName_to_CaseCollectionArgs = CaseCollectionName_Info['CaseCollectionName_to_CaseCollectionArgs']

        # update Case Args Settings
        self.SPACE = SPACE
        self.Case_Args_Settings = Case_Args_Settings
        self.TagRec_to_TagRecArgs = Case_Args_Settings.get('TagRec_to_TagRecArgs', {})  
        self.FltName_to_FltArgs = Case_Args_Settings.get('FltName_to_FltArgs', {}) 

        self.dataset_init = dataset_init


    def setup_triggerfn(self, TriggerFn = None):
        SPACE = self.SPACE
        TriggerCaseBase_Args = self.TriggerCaseBase_Args
        TriggerName = TriggerCaseBase_Args['Trigger']['TriggerName']
        if TriggerFn is None:
            TriggerFn = CaseSet_Trigger_Fn(TriggerName, SPACE)
        self.TriggerFn = TriggerFn


    def load_CaseCollection(self, 
                            CaseCollectionName, 
                            CaseCollectionPath, 
                            Trigger_Args,
                            ):
        logger.info(f'Load CaseCollection {CaseCollectionName} from {CaseCollectionPath}')
        # self.CaseSetName_to_caseset = datasets.load_from_disk(CaseCollectionPath)
        # return self.CaseSetName_to_caseset
        

        CaseName_list = os.listdir(CaseCollectionPath)
        CaseSetName_to_caseset = {}
        for CaseName in CaseName_list:
            if 'T.' != CaseName[:2]: continue
            CasePath = os.path.join(CaseCollectionPath, CaseName)
            # caseset = datasets.load_from_disk(CasePath)
            df_case_ids_path = os.path.join(CasePath, 'df_case_ids.p')
            df_case = pd.read_pickle(df_case_ids_path)
            # caseset_metainfo = pd.read_pickle(os.path.join(CasePath, 'caseset_metainfo.p'))
            with open(os.path.join(CasePath, 'caseset_metainfo.json'), 'r') as f:
                caseset_metainfo = json.load(f)

            #### TODO: add some assertations here.

            caseset = CaseSet(
                onecohort_record_base = self.onecohort_record_base,
                Trigger_Args = Trigger_Args,
                caseset_metainfo = caseset_metainfo,
                df_case = df_case, 
                # Case_Args_Settings = self.Case_Args_Settings,
                Case_Proc_Config = self.Case_Proc_Config,
            )

            CaseSetName_to_caseset[CaseName] = caseset
        
        self.CaseSetName_to_caseset = CaseSetName_to_caseset
        logger.info(f'======== Load CaseCollection  <{CaseCollectionName}>  Done! =========')
        
        return CaseSetName_to_caseset
    

    # ----------------- trigger task is most important -----------------
    def execute_triggertask(self, 
                            dataset = None, 
                            column_names = None, 
                            onecohort_record_base = None,
                            CaseCollectionName = None,  
                            CaseCollectionArgs = None, 
                            TriggerCaseBase_Args = None, 
                            Case_Proc_Config = None):


        if self.dataset_init is not None:
            dataset = self.dataset_init

        if onecohort_record_base is None: 
            onecohort_record_base = self.onecohort_record_base


        if not hasattr(self, 'TriggerFn'): 
            self.setup_triggerfn()
        if Case_Proc_Config is None: 
            Case_Proc_Config = self.Case_Proc_Config
        if TriggerCaseBase_Args is None: 
            TriggerCaseBase_Args = self.TriggerCaseBase_Args
        if CaseCollectionArgs is None:
            CaseCollectionArgs = TriggerCaseBase_Args['Trigger']
        if CaseCollectionName is None:
            CaseCollectionName = self.CaseCollectionName_list[0]

        TriggerFn = self.TriggerFn
        Trigger_Args = TriggerFn.Trigger_Args
        CaseCollectionPath = os.path.join(self.cohort_casebase_folder, CaseCollectionName)
        
        
        if Case_Proc_Config.get('load_casecollection', True) == True and os.path.exists(CaseCollectionPath):
            CaseSetName_to_caseset = self.load_CaseCollection(CaseCollectionName, CaseCollectionPath, Trigger_Args)
            # should be the first CaseSetName_to_caseset
            self.CaseSetName_to_caseset = CaseSetName_to_caseset
            return CaseSetName_to_caseset 
        

        # ------------------------- part 1: ds_case trigger -------------------------
        Case_Proc_Config = self.Case_Proc_Config
        max_case_num = Case_Proc_Config.get('max_trigger_case_num', None)
        
        
        TriggerName = CaseCollectionArgs['TriggerName']
        get_CaseTrigger_from_RecordBase = TriggerFn.get_CaseTrigger_from_RecordBase
        
        df_case = get_CaseTrigger_from_RecordBase(onecohort_record_base, Trigger_Args)

        if dataset is not None:
            if isinstance(dataset, pd.DataFrame):
                # df_case = dataset
                if column_names is None:
                    column_names = dataset.columns
                    column_names = [i for i in column_names if '--' not in i]
                df_case_provided = dataset[column_names]
            else:
                if column_names is None:
                    column_names = dataset.column_names
                    column_names = [i for i in column_names if '--' not in i]
                df_case_provided = dataset.select_columns(column_names).to_pandas()

            case_id_columns = Trigger_Args['case_id_columns']
            logger.info(f'the initial triggered df_case.shape:       {df_case.shape}')
            logger.info(f'the df_case_provided shape:                {df_case_provided.shape}')
            df_case = pd.merge(df_case[case_id_columns], df_case_provided, on = case_id_columns, how = 'inner')
            logger.info(f'the triggered df_case.shape after merging: {df_case.shape}')

            if len(df_case) == 0:
                raise ValueError(f'The final df_case is zero. Please check the dataset and the trigger arguments.')

        logger.info(f'Trigger {TriggerName} is done: {df_case.shape}')
        if max_case_num is not None: df_case = df_case.iloc[:max_case_num]

        # ----------------------- part 2: ds_case rectag -----------------------
        TagRec_list = CaseCollectionArgs.get('TagRec', [])
        TagRec_to_TagRecArgs = self.TagRec_to_TagRecArgs
        cohort_casebase_folder = self.cohort_casebase_folder
        for TagRecName in TagRec_list:
            # execute_tagrectask will consider the saving probelm. 
            logger.info(f'Execute TagRec Task: {TagRecName}')
            df_case = execute_tagrectask(
                df_case,
                onecohort_record_base, 
                TagRecName,
                cohort_casebase_folder,
                TagRec_to_TagRecArgs, 
                Case_Proc_Config
            )

        # ----------------------- part 3: filter data ---------------------
        FilterName = CaseCollectionArgs.get('Filter', None)
        if FilterName is None:
            logger.info(f'No Filter is applied. Keep All')
            df_case['Selected'] = 1
            # return df_case
        else:
            FilterTagging = self.FltName_to_FltArgs[FilterName]
            logger.info(f'Select df_case with {FilterName} and args: {FilterTagging}')
            logical_operator = 'and' if '-OR' not in FilterName else 'or'
            if len(FilterTagging) == 0:
                df_case['Selected'] = 1
            else:
                # logger.info(f'\n\n\nlogical_operator: {logical_operator}\n\n\n')
                logger.info(f'before filtering: {df_case.shape}')
                df_case['Selected'] = apply_multiple_conditions(df_case, 
                                                                FilterTagging, 
                                                                logical_operator=logical_operator).astype(int)
                
                df_case = df_case[df_case['Selected'] == 1].reset_index(drop = True)   
            logger.info(f'Filtering df_case is done: {df_case.shape}')


        # ----------------------- part 4: creating the CaseSetName_to_caseset -----------------------
        GroupMethodName = CaseCollectionArgs.get('Group', 'Base')
        assert type(df_case) == pd.DataFrame
        # GroupName_to_dfcase = {GroupMethodName: df_case}
        CaseSetName_to_caseset = {}
        caseset_chunk_size = Case_Proc_Config['caseset_chunk_size']
        SPACE = self.SPACE


        caseset_meta_template = {
            'cohort_casebase_folder': cohort_casebase_folder.replace(SPACE.get('DATA_CASE', '--'), '$DATA_CASE$'), 
            'CaseCollectionName': CaseCollectionName,
            'max_case_num': max_case_num,
            'GroupName': 'Base', 
            'caseset_chunk_size': caseset_chunk_size,
        }

        assert len(df_case) > 0, f'df_case is empty'

        df_small_list, Case_Proc_Config = split_dataframe(df_case, Case_Proc_Config)
        caseset_chunk_size_updated = Case_Proc_Config['max_chunk_size_updated']
        for idx, df_small in enumerate(df_small_list):
            if len(df_small) == 0: continue 

            # update caseset_metainfo
            caseset_metainfo = caseset_meta_template.copy()
            # idx_start = caseset_chunk_size * idx
            caseset_metainfo['idx'] = idx
            caseset_metainfo['caseset_chunk_size'] = caseset_chunk_size_updated

            caseset = CaseSet(
                onecohort_record_base = self.onecohort_record_base,
                Trigger_Args = Trigger_Args,
                caseset_metainfo = caseset_metainfo,
                df_case = df_small, 
                # Case_Args_Settings = self.Case_Args_Settings,
                Case_Proc_Config = self.Case_Proc_Config,
                SPACE = self.SPACE,
            )
            CaseSetName = caseset.CaseSetName
            CaseSetName_to_caseset[CaseSetName] = caseset

            if Case_Proc_Config['save_data'] == True:
                os.makedirs(caseset.caseset_path, exist_ok = True)
                path_df_case = os.path.join(caseset.caseset_path, 'df_case_ids.p')
                
                if os.path.exists(path_df_case):
                    logger.info(f'df_case exists in {path_df_case}')
                    df_case_from_disk = pd.read_pickle(path_df_case)
                    # Verify that PID and ObsDT are the same between disk and current data
                    # print(df_case_from_disk.columns)
                    # print(caseset.df_case.columns)
                    # print(df_case_from_disk.shape)
                    # print(caseset.df_case.shape)

                    df_case_from_disk = df_case_from_disk.reset_index(drop = True)
                    caseset.df_case = caseset.df_case.reset_index(drop = True)

                    pid_match = (df_case_from_disk['PID'] == caseset.df_case['PID']).all()
                    obsdt_match = (df_case_from_disk['ObsDT'] == caseset.df_case['ObsDT']).all()
                    assert pid_match and obsdt_match, f"PID or ObsDT mismatch between stored and current data"
                    
                    if '_keep_ratio' in df_case_from_disk.columns:  
                        assert '_keep_ratio' in caseset.df_case.columns, f"No _keep_ratio column in caseset.df_case"
                        keep_ratio_match = (df_case_from_disk['_keep_ratio'] == caseset.df_case['_keep_ratio']).all()
                        assert keep_ratio_match, f"keep_ratio mismatch between stored and current data"
                        logger.info(f'df_case is the same as the one on disk for _keep_ratio column')
                    else:
                        caseset.df_case.to_pickle(path_df_case) 
                        logger.info(f'Save df_case of size {caseset.df_case.shape} to {path_df_case} with _keep_ratio column')
                    # logger.info(f'Save df_case of size {caseset.df_case.shape} to {path_df_case}')
                    logger.info(f'df_case is the same as the one on disk')
                else:
                    caseset.df_case.to_pickle(path_df_case)
                    logger.info(f'Save df_case of size {caseset.df_case.shape} to {path_df_case}')

                path_case_metainfo = os.path.join(caseset.caseset_path, 'caseset_metainfo.json')
                if os.path.exists(path_case_metainfo):
                    logger.info(f'caseset_metainfo exists in {path_case_metainfo}, skip saving')
                else:
                    with open(path_case_metainfo, 'w') as f:
                        json.dump(caseset_metainfo, f, indent=4)
                    logger.info(f'Save caseset_metainfo to {path_case_metainfo}')
                

        # self.CaseSetRepo[CaseCollectionName] = CaseSetName_to_caseset
        logger.info(f'Trigger Task {TriggerName} is done: {len(CaseSetName_to_caseset)} casesets. (No CFs yet)')

        self.CaseSetName_to_caseset = CaseSetName_to_caseset
        return CaseSetName_to_caseset 
    

    def execute_filterCaseSetTask(self, 
                                  CaseCollectionName = None,  
                                  CaseCollectionArgs = None, ):
    
        # FltName, 
        # FltTagging
        TriggerFn = self.TriggerFn
        Trigger_Args = TriggerFn.Trigger_Args
        CaseCollectionPath = os.path.join(self.cohort_casebase_folder, CaseCollectionName)
        
        
        Case_Proc_Config = self.Case_Proc_Config
        CaseSetName_to_caseset = self.CaseSetName_to_caseset
        if Case_Proc_Config['load_casecollection'] == True and os.path.exists(CaseCollectionPath):

            # distroy the original CaseSetName_to_caseset if any
            if hasattr(self, 'CaseSetName_to_caseset'):
                for CaseSetName, caseset in self.CaseSetName_to_caseset.items():
                    del caseset.df_case
                    del caseset.ds_case
                del self.CaseSetName_to_caseset

            CaseSetName_to_caseset = self.load_CaseCollection(CaseCollectionName, CaseCollectionPath, Trigger_Args)
            # should be the first CaseSetName_to_caseset
            self.CaseSetName_to_caseset = CaseSetName_to_caseset

            return CaseSetName_to_caseset 
        
        
        FltName = CaseCollectionArgs['Filter']
        FltTagging = self.FltName_to_FltArgs[FltName]

        logger.info('======== Filter CaseSet Task for New CaseSetName_to_CaseSet =========')
        
        logger.info(f'Filtering df_case with {FltName}: {FltTagging}')

        TriggerFn = self.TriggerFn
        Trigger_Args = TriggerFn.Trigger_Args

        CaseSetName_to_caseset_New = {}
        for CaseSetName, caseset in CaseSetName_to_caseset.items():
            ds_case = caseset.ds_case 
            df_case_orig = caseset.df_case
            tag_columns = [i for i in ds_case.column_names if '--' not in i and i not in df_case_orig.columns]
            print('\n\nLook Here ========== {}\n\n'.format(len(tag_columns)))
            print('\n\nLook Here ========== {}\n\n'.format(tag_columns))
            df_case = ds_case.select_columns(tag_columns).to_pandas()


            df_case = pd.concat([df_case_orig, df_case], axis = 1)
            caseset.df_case = df_case
            assert len(caseset.df_case) == len(caseset.ds_case), f'len(caseset.df_case) != len(df_case)'
            print('\n\nLook Here ========== {}\n\n'.format(df_case.columns))

            # if '_keep_ratio' not in df_case.columns:
            #     df_case['_keep_ratio'] = np.random.rand(len(df_case))
            #     df_case.to_pickle(os.path.join(caseset.caseset_path, 'df_case_ids.p'))
            #     logger.info(f'\n\ndf_case has no _keep_ratio column, add one.\n\n')
            # else:
            #     logger.info(f'\n\ndf_case already has _keep_ratio column, skip saving\n\n')

            logger.info(f'before filtering: {df_case.shape}')

            logical_operator = 'and' if '-OR' not in FltName else 'or'
            # logger.info(f'\n\n\nlogical_operator: {logical_operator}\n\n\n')
            df_case['Selected'] = apply_multiple_conditions(df_case, 
                                                            FltTagging, 
                                                            logical_operator=logical_operator).astype(int)
            
            # df_case['Selected'] = apply_multiple_conditions(df_case, FltTagging, logical_operator='and').astype(int)
            df_case_selected = df_case[df_case['Selected'] == 1].reset_index(drop = True)
            logger.info(f'after filtering: {df_case_selected.shape}')

            if len(df_case_selected) == 0:
                logger.info(f'df_case_selected is empty, skip generating casesets')
                continue


            

            caseset_metainfo = caseset.caseset_metainfo.copy()

            CaseCollectionName = caseset_metainfo['CaseCollectionName'] + f'-{FltName}'
            # print(CaseCollectionName, self.CaseCollectionName_list)
            assert CaseCollectionName in self.CaseCollectionName_list
            caseset_metainfo['CaseCollectionName'] = CaseCollectionName


            # don't need to save the original df_case
            del caseset.df_case
            del caseset.ds_case 


            print(df_case_selected.columns)
            print(df_case_selected.dtypes)
            caseset = CaseSet(
                onecohort_record_base = self.onecohort_record_base,
                Trigger_Args = Trigger_Args,
                caseset_metainfo = caseset_metainfo,
                df_case = df_case_selected, 

                # Case_Args_Settings = self.Case_Args_Settings,
                Case_Proc_Config = self.Case_Proc_Config,
            ) 

        
            CaseSetName = caseset.CaseSetName

            Case_Proc_Config = self.Case_Proc_Config
            if Case_Proc_Config['save_data'] == True:
                os.makedirs(caseset.caseset_path, exist_ok = True)

                path_df_case = os.path.join(caseset.caseset_path, 'df_case_ids.p')
                if os.path.exists(path_df_case):
                    logger.info(f'df_case exists in {path_df_case}, skip saving')
                else:
                    caseset.df_case.to_pickle(path_df_case)
                    logger.info(f'Save df_case of size {caseset.df_case.shape} to {path_df_case}')

                path_case_metainfo = os.path.join(caseset.caseset_path, 'caseset_metainfo.json')
                if os.path.exists(path_case_metainfo):
                    logger.info(f'caseset_metainfo exists in {path_case_metainfo}, skip saving')
                else:
                    with open(path_case_metainfo, 'w') as f:
                        json.dump(caseset_metainfo, f, indent=4)
                    logger.info(f'Save caseset_metainfo to {path_case_metainfo}')

            CaseSetName_to_caseset_New[CaseSetName] = caseset

        del self.CaseSetName_to_caseset

        # self.CaseSetRepo[CaseCollectionName] = CaseSetName_to_caseset_New
        CaseSetName_to_caseset = CaseSetName_to_caseset_New
        self.CaseSetName_to_caseset = CaseSetName_to_caseset
        return CaseSetName_to_caseset
    

    def execute_casefn_task(self, CF_list, CaseSetName_to_caseset = None):

        if CaseSetName_to_caseset is None:
            CaseSetName_to_caseset = self.CaseSetName_to_caseset

        Case_Proc_Config = self.Case_Proc_Config

        onecohort_record_base = self.onecohort_record_base
        Case_Args_Settings = self.Case_Args_Settings

        # logger.info(f'\n\n\nExecute CaseFnTask with {CaseFnTaskArgs}\n\n\n')
        casefn_task = CaseFn_Task(
            CaseSetName_to_caseset = CaseSetName_to_caseset,
            onecohort_record_base = onecohort_record_base,
            CF_list = CF_list,
            Case_Proc_Config = Case_Proc_Config,
            SPACE = self.SPACE,
        )
        n_cpus = Case_Proc_Config['n_cpus'] 
        if n_cpus > 1:
            CaseSetName_to_caseset = casefn_task.execute_casefn_tasks_with_multiple_process()
        else:
            CaseSetName_to_caseset = casefn_task.execute_casefn_tasks()
        self.CaseSetName_to_caseset = CaseSetName_to_caseset
        return CaseSetName_to_caseset


    def init_OneCohortTrigger(self):

        # 1. setup trigger function
        self.setup_triggerfn()
        TriggerFn = self.TriggerFn
        Trigger_Args = TriggerFn.Trigger_Args

        # 2. CaseCollectionName_to_CaseCollectionArgs
        CaseCollectionName_to_CaseCollectionArgs = self.CaseCollectionName_to_CaseCollectionArgs
        CaseCollectionName_list = self.CaseCollectionName_list


        # 3. load the last CaseCollection
        CaseSetName_to_caseset = None 
        index = 0
        for CaseCollectionName in reversed(CaseCollectionName_list):
            CaseeCollectionFolder = os.path.join(self.cohort_casebase_folder, CaseCollectionName)
            if os.path.exists(CaseeCollectionFolder) and self.Case_Proc_Config['load_casecollection'] == True:
                CaseSetName_to_caseset = self.load_CaseCollection(CaseCollectionName, CaseeCollectionFolder, Trigger_Args)
                index = CaseCollectionName_list.index(CaseCollectionName)
                break


        # 4. start from the beginning or continue from the last CaseCollection
        CaseCollectionName_list_to_collect = CaseCollectionName_list[index:]  
        assert len(CaseCollectionName_list_to_collect) > 0

        if CaseSetName_to_caseset is None: 
            logging.info(f'No CaseCollection is loaded, start from the Triggering.')
        else:
            logging.info(f'CaseCollection {CaseCollectionName} is loaded')

        logging.info('processing CaseFnTask for the followings:')
        for i in CaseCollectionName_list_to_collect:
            logging.info(f'\t{i}')

        for CaseCollectionName in CaseCollectionName_list_to_collect:
            CaseCollectionArgs = CaseCollectionName_to_CaseCollectionArgs[CaseCollectionName]
            if 'TriggerName' in CaseCollectionArgs:
                logging.info('===============================================')
                logging.info(f'Execute Trigger Task: {CaseCollectionName}')
                
                if CaseSetName_to_caseset is None:
                    CaseSetName_to_caseset = self.execute_triggertask(
                        CaseCollectionName = CaseCollectionName, 
                        CaseCollectionArgs = CaseCollectionArgs
                    )

                CF_list = CaseCollectionArgs.get('CaseFnTasks', None)
                if CF_list is not None:
                    CaseSetName_to_caseset = self.execute_casefn_task(
                        CF_list = CF_list, 
                        CaseSetName_to_caseset = CaseSetName_to_caseset
                    )
                logging.info('================= Done =========================\n\n')
            else:
                logging.info('===============================================')
                logging.info(f'Execute FilterCaseSet Task: {CaseCollectionName}')
                # FilterCaseSet
                CaseSetName_to_caseset = self.execute_filterCaseSetTask(
                    CaseCollectionName = CaseCollectionName,
                    CaseCollectionArgs = CaseCollectionArgs
                )
                CF_list = CaseCollectionArgs.get('CaseFnTasks', None)
                if CF_list is not None:
                    CaseSetName_to_caseset = self.execute_casefn_task(
                        CF_list = CF_list, 
                        CaseSetName_to_caseset = CaseSetName_to_caseset
                    )
                logging.info('================= Done =========================\n\n')

        self.CaseSetName_to_caseset = CaseSetName_to_caseset
        return CaseSetName_to_caseset


class Case_Base(Base):
    def __init__(self, 
                 record_base, 
                 TriggerCaseBaseName_to_CohortNameList, #  = None, 
                 TriggerCaseBaseName_to_TriggerCaseBaseArgs = None,
                 TriggerCaseBaseName_to_dataset_init = None,
                 Case_Proc_Config = None, 
                 Case_Args_Settings = None, 
                 CaseSettingInfo = None, 
                 ):
        '''
        CaseBase_Args is OneCohort_CaseBase_Args
        '''
        self.record_base = record_base
        self.Case_Proc_Config = Case_Proc_Config
        self.Case_Args_Settings = Case_Args_Settings

        self.TriggerCaseBaseName_to_CohortNameList      = TriggerCaseBaseName_to_CohortNameList
        self.TriggerCaseBaseName_to_TriggerCaseBaseArgs = TriggerCaseBaseName_to_TriggerCaseBaseArgs
        self.TriggerCaseBaseName_to_dataset_init = TriggerCaseBaseName_to_dataset_init
        self.TriggerCaseBaseName_to_CaseSetNameToCaseset = {}
        self.TriggerCaseBaseName_to_CFtoCFvocab = {}
        self.CaseSettingInfo = CaseSettingInfo

        logger.info(f'Init CaseBase\n')
        for TriggerCaseBaseName, CohortName_list in TriggerCaseBaseName_to_CohortNameList.items():
            d = {}
            for CohortName in CohortName_list:
                logger.info(f'[start] ===========================================\n')
                # assert CohortName in HumanRecordRecfeat_Args
                onecohort_record_base = record_base.CohortName_to_OneCohortRecordBase[CohortName]
                TriggerCaseBase_Args = TriggerCaseBaseName_to_TriggerCaseBaseArgs[TriggerCaseBaseName]
    
                if TriggerCaseBaseName_to_dataset_init is None:
                    dataset_init = None
                    dataset_init_name = None
                else:
                    dataset_init_info = TriggerCaseBaseName_to_dataset_init[TriggerCaseBaseName]
                    dataset_init = dataset_init_info['dataset_init']
                    dataset_init_name = dataset_init_info['dataset_init_name']
                    
                onecohort_trigger_casebase = OneCohortTrigger_CaseBase(
                    onecohort_record_base = onecohort_record_base, 
                    TriggerCaseBase_Args  = TriggerCaseBase_Args,
                    Case_Proc_Config      = Case_Proc_Config,
                    Case_Args_Settings    = Case_Args_Settings,
                    SPACE                 = record_base.SPACE,
                    dataset_init          = dataset_init,
                    dataset_init_name     = dataset_init_name,
                )
                onecohort_trigger_casebase.init_OneCohortTrigger()
                for CaseSetName, caseset in onecohort_trigger_casebase.CaseSetName_to_caseset.items():
                    d['C.' + CohortName + ':' + CaseSetName] = caseset
                logger.info(f'[end] ===========================================\n\n')
            
            self.TriggerCaseBaseName_to_CaseSetNameToCaseset[TriggerCaseBaseName] = d


            if hasattr(onecohort_trigger_casebase, 'CaseFnTaskArgs'):
                CaseFnTaskArgs = onecohort_trigger_casebase.CaseFnTaskArgs
                CF_to_CFInfo = CaseFnTaskArgs['ObsName_To_ObsInfo'].CF_to_CFInfo
                CF_to_CFvocab = {CF: CF_to_CFInfo[CF]['CFvocab'] for CF in CF_to_CFInfo}  
                self.TriggerCaseBaseName_to_CFtoCFvocab[TriggerCaseBaseName] = CF_to_CFvocab


        try:
            Record_Proc_Config = record_base.Record_Proc_Config
        except:
            Record_Proc_Config = None


        self.config = {
            # 'CF_to_CFvocab': self.CF_to_CFvocab,
            # 'Name_to_Case': self.Name_to_Case,

            'CohortName_list': CohortName_list,
            'CohortName_to_OneCohortArgs': record_base.CohortName_to_OneCohortArgs,
            'CaseSettingInfo': CaseSettingInfo,

            'Record_Proc_Config': Record_Proc_Config,
            'Case_Proc_Config': Case_Proc_Config,
            
            'TriggerCaseBaseName': TriggerCaseBaseName,
            'TriggerCaseBaseName_to_CohortNameList': TriggerCaseBaseName_to_CohortNameList,
            'TriggerCaseBaseName_to_TriggerCaseBaseArgs': TriggerCaseBaseName_to_TriggerCaseBaseArgs,
            
            # 'OneAIDataName': OneAIDataName, 
            # 'OneEntryArgs': OneEntryArgs,
            
            'SPACE': record_base.SPACE,

        }
        self.SPACE = record_base.SPACE


    @property
    def Name_to_Case(self):
        # casebase_names = [i for i in self.TriggerCaseBaseName_to_CaseSetNameToCaseset]
        li_name_to_case = [d for d in self.TriggerCaseBaseName_to_CaseSetNameToCaseset.values()]
        Name_to_Case = {k: v for d in li_name_to_case for k, v in d.items()}
        return Name_to_Case


    @property
    def CF_to_CFvocab(self):
        CF_to_CFvocab_final = {}
        SPACE = self.SPACE

        Name_to_Case = self.Name_to_Case
        for k, v in Name_to_Case.items():
            CF_list = list(set([i.split('--')[0] for i in v.ds_case.column_names if '--tid' in i]))
            CF_fn_list = [Case_Fn(CF, SPACE) for CF in CF_list]
            CF_to_CFvocab = {CF: CF_fn.COVocab for CF, CF_fn in zip(CF_list, CF_fn_list)}
            CF_to_CFvocab_final.update(CF_to_CFvocab)
            break 

        return CF_to_CFvocab_final
    

    def create_dataset(self):
        Name_to_Case = self.Name_to_Case
        CF_to_CFvocab = self.CF_to_CFvocab

        df_list = []
        ds_list = []
        for k, v in Name_to_Case.items():
            # ds_case_tag  = datasets.Dataset.from_pandas(v.df_case)
            df_case_tag =  v.df_case
            ds_case_data = v.ds_case
            ds_case_total = ds_case_data

            print(k, ds_case_total)

            if len(df_case_tag) > 0:
                df_list.append(df_case_tag)
                ds_list.append(ds_case_total)
            else:
                logger.info(f'{k} has no cases, skip')

        # 1. concatenate df_list
        df_total_tag = pd.concat(df_list).reset_index(drop=True)

        if type(ds_list[0]) == datasets.Dataset:
            ds_total_data = datasets.concatenate_datasets(ds_list)
        elif type(ds_list[0]) == pd.DataFrame:
            ds_total_data = pd.concat(ds_list).reset_index(drop=True)
            ds_total_data = datasets.Dataset.from_pandas(ds_total_data)
            assert len(ds_total_data) == len(df_total_tag)
        else:
            raise ValueError(f'ds_list[0] is not a pandas DataFrame or a datasets.Dataset')

        df_case_tag_ids = df_total_tag[['PID', 'ObsDT']]
        df_case_data_ids = ds_total_data.select_columns(['PID', 'ObsDT']).to_pandas()
        assert df_case_tag_ids.equals(df_case_data_ids), "PID and ObsDT don't match between tag and data datasets"

        ds_total_tag = datasets.Dataset.from_pandas(df_total_tag.drop(columns = ['PID', 'ObsDT']))

        # Check for duplicated columns between ds_total_tag and ds_total_data
        tag_columns = set(ds_total_tag.column_names)
        data_columns = set(ds_total_data.column_names)
        duplicated_columns = tag_columns.intersection(data_columns)
        
        if duplicated_columns:
            logger.warning(f"Found duplicated columns: {duplicated_columns}")
            # Remove duplicated columns from ds_total_data
            ds_total_data = ds_total_data.remove_columns(duplicated_columns)
            logger.info(f'ds_total_data after removing duplicated columns: {ds_total_data.column_names}')
            
        ds_case_total = datasets.concatenate_datasets([ds_total_data, ds_total_tag], axis = 1)
        dataset = ds_case_total

        # config 
        config = self.config
        config['CF_to_CFvocab'] = CF_to_CFvocab
        dataset_info = DatasetInfo.from_dict({'config_name': config})
        dataset.info.update(dataset_info)

        return dataset
    

    def create_df_case_tag(self):
        Name_to_Case = self.Name_to_Case
        df_case = pd.concat([v.df_case for v in Name_to_Case.values()], axis = 0).reset_index(drop = True)
        # df_case.columns
        return df_case