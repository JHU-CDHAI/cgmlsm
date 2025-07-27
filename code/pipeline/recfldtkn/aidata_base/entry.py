import os
import copy
import json 
import itertools
import logging
import datasets
import numpy as np 
import pandas as pd
from pprint import pprint 
from datetime import datetime
from deepdiff import DeepDiff
from datasets import DatasetInfo
from ..base import apply_multiple_conditions, assign_caseSplitTag_to_dsCase
from ..base import Base
from ..case_base.case_base import CASESET_TRIGGER_PATH
from ..case_base.casefnutils.casefn import Case_Fn #  import AIDATA_ENTRYINPUT_PATH
        

logger = logging.getLogger(__name__)


AIDATA_SPLIT_PATH  = 'fn/fn_aidata/split'
AIDATA_ENTRYINPUT_PATH  = 'fn/fn_aidata/entryinput'
AIDATA_ENTRYOUTPUT_PATH = 'fn/fn_aidata/entryoutput'
FILTER_COLUMN = 'Filter'
SPLIT_COLUMN = 'Split' 
SPLIT_SUBCOLUMN = 'SplitSub'  



class EntryForIO(Base):

    def update_entry_for_inference(self):
        OneEntryArgs = self.OneEntryArgs
        del OneEntryArgs['Split_Part']
        del OneEntryArgs['Output_Part']
        self.OneEntryArgs = OneEntryArgs


    def get_CFs_ForInference(self):
        # OneAIDataArgs = self.OneAIDataArgs
        OneEntryArgs = self.OneEntryArgs # OneAIDataArgs['OneEntryArgs']
        Input_Part = OneEntryArgs['Input_Part']
        
        if 'InputCFs_ForInference' not in Input_Part:
            InputCFs_Args = Input_Part['CF_list']
            if type(InputCFs_Args) == dict:
                CFs_ForInference = list(set(sum(list(InputCFs_Args.values()), [])))
            elif type(InputCFs_Args) == list:
                CFs_ForInference = InputCFs_Args
            else:
                raise ValueError(f'InputCFs_Args should be either dict or list, but got {type(InputCFs_Args)}') 
        else:
            CFs_ForInference = Input_Part['InputCFs_ForInference']
            assert type(CFs_ForInference) == list   

        logger.info(f'CFs_ForInference: {CFs_ForInference}')
        return CFs_ForInference


    def save_entry(self, data_path):
        entry_config = {}
        for attr in ['OneEntryArgs', 'CF_Config']:
            if hasattr(self, attr):
                entry_config[attr] = getattr(self, attr)
                # setattr(self, attr, None)

        entry_config_path = os.path.join(data_path, 'EntryConfig.json')
        with open(entry_config_path, 'w') as f:
            json.dump(entry_config, f, indent=4)

    @classmethod
    def load_entry(cls, data_path, SPACE):
        with open(os.path.join(data_path, 'EntryConfig.json'), 'r') as f:
            entry_config = json.load(f)

        logger.info(f'load entry from {data_path}')
        return cls(SPACE = SPACE,**entry_config)
    

class EntryForInputOutputOperation(Base):

    def get_CFList_for_AIData(self, OneEntryArgs = None):
        
        if OneEntryArgs is None: OneEntryArgs = self.OneEntryArgs
        # Input_Part  = OneEntryArgs['Input_Part']
        # Output_Part = OneEntryArgs.get('OneEntryArgs', {})

        try:
            INPUT_CFs = self.get_INPUT_CFs(OneEntryArgs)
        except:
            INPUT_CFs = []

        try: # TODO: remove this try-except
            OUTPUT_CFs = self.get_OUTPUT_CFs(OneEntryArgs)
        except:
            OUTPUT_CFs = []
        TotalCFList = INPUT_CFs + OUTPUT_CFs
        self.TotalCFList = TotalCFList
        return TotalCFList 
    


    def load_trigger_path(self, OneEntryArgs = None, SPACE = None):
        if OneEntryArgs is None: OneEntryArgs = self.OneEntryArgs
        if SPACE is None: SPACE = self.SPACE 

        TriggerName = self.TriggerName
        pypath_for_trigger = os.path.join(SPACE['CODE_FN'], CASESET_TRIGGER_PATH, TriggerName + '.py')
        self.pypath_for_trigger = pypath_for_trigger
        if self.pypath_for_trigger is not None and os.path.exists(self.pypath_for_trigger):
            logger.info(f'Load Trigger Method for {self.pypath_for_trigger}')  
            module = self.load_module_variables(pypath_for_trigger)
            self.TriggerName = TriggerName
            self.Trigger_Args = module.Trigger_Args
            self.case_id_columns = self.Trigger_Args['case_id_columns']
        else:
            logger.warning(f'No Trigger Method for {self.pypath_for_trigger}')


    def load_entry_pypath(self, OneEntryArgs = None, SPACE = None):
        if OneEntryArgs is None: OneEntryArgs = self.OneEntryArgs
        if SPACE is None: SPACE = self.SPACE 

        # --------- input_part ----------
        Input_Part = OneEntryArgs.get('Input_Part', {})
        EntryInputMethod = Input_Part.get('EntryInputMethod')
        pypath_for_entryinput = os.path.join(SPACE['CODE_FN'],  AIDATA_ENTRYINPUT_PATH, f'{EntryInputMethod}.py')
        self.pypath_for_entryinput = pypath_for_entryinput 
        if self.pypath_for_entryinput is not None and os.path.exists(self.pypath_for_entryinput):
            logger.info(f'Load EntryInput Method for {self.pypath_for_entryinput}')  
            module = self.load_module_variables(self.pypath_for_entryinput)
            self.tfm_fn_AIInputData   = module.tfm_fn_AIInputData
            self.entry_fn_AIInputData = module.entry_fn_AIInputData
            self.get_INPUT_CFs = module.get_INPUT_CFs
        else:
            logger.warning(f'No EntryInput Method for {self.pypath_for_entryinput}')

        # --------- output_part -----------
        Output_Part = OneEntryArgs.get('Output_Part', {})
        if len(Output_Part) > 0: 
            EntryOutputMethod = Output_Part.get('EntryOutputMethod')
            pypath_for_entryoutput = os.path.join(SPACE['CODE_FN'],  AIDATA_ENTRYOUTPUT_PATH, f'{EntryOutputMethod}.py')
            self.pypath_for_entryoutput = pypath_for_entryoutput
            if self.pypath_for_entryoutput is not None and os.path.exists(self.pypath_for_entryoutput): 
                logger.info(f'Load EntryOutput Method for {self.pypath_for_entryoutput}')  
                module = self.load_module_variables(self.pypath_for_entryoutput)
                self.get_OUTPUT_CFs = module.get_OUTPUT_CFs
                self.entry_fn_AITaskData = module.entry_fn_AITaskData
                # TaskUpdate_fn = module.TaskUpdate_fn
                # if TaskUpdate_fn is not None:
                #     logger.info(f'Update Task Args for {self.EntryOutputMethod}')   
                #     self.Tasks_FullArgs = TaskUpdate_fn(self.Tasks_FullArgs)
            else:
                logger.warning(f'No EntryOutput Method for {self.pypath_for_entryoutput}')

        Split_Part = OneEntryArgs.get('Split_Part', {})
        if len(Split_Part) > 0: 
            SplitMethod = Split_Part.get('SplitMethod')
            pypath_for_split = os.path.join(SPACE['CODE_FN'],  AIDATA_SPLIT_PATH, f'{SplitMethod}.py')
            self.pypath_for_split = pypath_for_split
            if self.pypath_for_split is not None and os.path.exists(self.pypath_for_split): 
                logger.info(f'Load Split Method for {self.pypath_for_split}')  
                module = self.load_module_variables(self.pypath_for_split)
                self.dataset_split_tagging_fn = module.dataset_split_tagging_fn
                # TaskUpdate_fn = module.TaskUpdate_fn
                # if TaskUpdate_fn is not None:
                #     logger.info(f'Update Task Args for {self.EntryOutputMethod}')   
                #     self.Tasks_FullArgs = TaskUpdate_fn(self.Tasks_FullArgs)
            else:
                logger.warning(f'No Split Method for {self.pypath_for_split}')



    def setup_EntryFn_to_Data(self, 
                              Data, # ds_case 
                              CF_to_CFvocab,
                              OneEntryArgs = None, 
                              tfm_fn_AIInputData = None, 
                              entry_fn_AIInputData = None, 
                              entry_fn_AITaskData = None, 
                            ):
        
        if OneEntryArgs is None: 
            OneEntryArgs = self.OneEntryArgs
        if entry_fn_AIInputData is None:
            entry_fn_AIInputData = self.entry_fn_AIInputData
        if tfm_fn_AIInputData is None:
            tfm_fn_AIInputData = self.tfm_fn_AIInputData
        if entry_fn_AITaskData is None and OneEntryArgs.get('Split_Part', None) is not None:
            entry_fn_AITaskData = self.entry_fn_AITaskData
            
        assert entry_fn_AIInputData is not None
        # logger.info([CF for CF in CF_to_CFvocab.keys()])
        logger.info('entry_fn_AIInputData: {}'.format(entry_fn_AIInputData))
        logger.info('tfm_fn_AIInputData: {}'.format(tfm_fn_AIInputData))
        logger.info('entry_fn_AITaskData: {}'.format(entry_fn_AITaskData))

        # ds_case = Data['ds_case'] 
        # print(type(ds_case))
        # print(ds_case.format['format_kwargs'].get('transform', 'No-transform'))  
        # if 'transform' in ds_case.format['format_kwargs']:
        #     format_kwargs = ds_case.format['format_kwargs']
        #     logger.info(f'delete transform from ds_case with {format_kwargs}')
        #     ds_case.format['format_kwargs']['transform'] = None
        # Data['ds_case'] = ds_case

        if entry_fn_AITaskData is None:
            logger.info('entry_fn_AIInputData is executed')
            Data = entry_fn_AIInputData(Data, 
                                          CF_to_CFvocab, 
                                          OneEntryArgs,
                                          tfm_fn_AIInputData)
        else:
            logger.info('entry_fn_AITaskData is executed')
            Data = entry_fn_AITaskData(Data, 
                                         CF_to_CFvocab, 
                                         OneEntryArgs,
                                         tfm_fn_AIInputData,
                                         entry_fn_AIInputData,
                                         )
        return Data
    

class EntryAIData_Builder(EntryForIO,
                          EntryForInputOutputOperation):

    def __init__(self, 
                 OneEntryArgs = None, 
                 CF_Config = None, 
                 SPACE = None):
        
        self.SPACE = SPACE
        if OneEntryArgs is None:
            OneEntryArgs = {}
        self.OneEntryArgs = OneEntryArgs
        # self.TriggerName  = TriggerName
        self.Input_Part   = OneEntryArgs.get('Input_Part', None)
        self.Output_Part  = OneEntryArgs.get('Output_Part', None)
        self.Task_Part    = OneEntryArgs.get('Task_Part',   None)
        self.load_entry_pypath()
        # self.load_trigger_path()
        
        if len(OneEntryArgs) > 0:
            self.TotalCFList = self.get_CFList_for_AIData(OneEntryArgs)
            try:
                self.INPUT_CFs   = self.get_INPUT_CFs(OneEntryArgs)
            except:
                self.INPUT_CFs = []
            
        if CF_Config is not None:
            self.CF_Config = CF_Config
            self.CF_to_CFvocab = CF_Config['CF_to_CFvocab']

            CF_DataName = CF_Config['TriggerCaseBaseName']
            TriggerCaseBaseArgs = CF_Config['TriggerCaseBaseName_to_TriggerCaseBaseArgs'][CF_DataName]
            TriggerName = TriggerCaseBaseArgs['Trigger']['TriggerName']

            logger.info(f'set up TriggerName: {TriggerName}')
            logger.info(f'set up CF_Config: {[i for i in CF_Config]}')
            self.TriggerName = TriggerName
            self.load_trigger_path()

    ## used it only in the train processing 
    def merge_one_cf_dataset(self, CF_DataName_list):
        SPACE = self.SPACE
        ds_list = []
        ref_config = None
        ref_column_names = None
        for i, CF_DataName in enumerate(CF_DataName_list):
            if 'DATA_CFDATA' in SPACE:
                path = os.path.join(SPACE['DATA_CFDATA'], CF_DataName)
            else:
                path = os.path.join(SPACE['DATA_AIDATA'], CF_DataName)
            ds = datasets.load_from_disk(path)
            # config = copy.deepcopy(ds.info.config.__dict__) if hasattr(ds.info, 'config') else {}
            config = ds.config_name 
            column_names = copy.deepcopy(ds.column_names)

            if i == 0:
                # ref_config = config
                ref_column_names = sorted(column_names)
            else:
                # config_diff = DeepDiff(ref_config, config, ignore_order=True)
                column_diff = DeepDiff(ref_column_names, sorted(column_names), ignore_order=False)

                # if config_diff:
                #     raise ValueError(f"[{CF_DataName}] config mismatch:\n{config_diff}")
                if column_diff:
                    raise ValueError(f"[{CF_DataName}] column names mismatch:\n{column_diff}")
            
            ds_list.append(ds)

        # pprint(config)
        dataset = datasets.concatenate_datasets(ds_list)

        CF_list = list(set([i.split('--')[0] for i in dataset.column_names if '--tid' in i]))
        CF_fn_list = [Case_Fn(CF, SPACE) for CF in CF_list]
        CF_to_CFvocab = {CF: CF_fn.COVocab for CF, CF_fn in zip(CF_list, CF_fn_list)}

        CF_DataName = config['TriggerCaseBaseName']
        TriggerCaseBaseArgs = config['TriggerCaseBaseName_to_TriggerCaseBaseArgs'][CF_DataName]
        TriggerName = TriggerCaseBaseArgs['Trigger']['TriggerName']

        logger.info(f'set up TriggerName: {TriggerName}')
        logger.info(f'set up CF_Config: {[i for i in config]}')
        config['CF_to_CFvocab'] = CF_to_CFvocab
        self.CF_Config = config
        self.CF_to_CFvocab = CF_to_CFvocab # full CF_list from cf_dataset. 
        self.TriggerName = TriggerName
        self.load_trigger_path()
        dataset_info = DatasetInfo.from_dict({'config_name': config})
        dataset.info.update(dataset_info)
        return dataset


    ## used it only in the train processing  
    def split_cf_dataset(self, dataset, OneEntryArgs = None):
        if OneEntryArgs is None: OneEntryArgs = self.OneEntryArgs
        dataset_split_tagging_fn = self.dataset_split_tagging_fn

        CF_list = list(set([i.split('--')[0] for i in dataset.column_names if '--tid' in i]))
        tag_columns = [i for i in dataset.column_names if '--' not in i]
        df_tag = dataset.select_columns(tag_columns).to_pandas()

        ##########################################################
        df_tag = dataset_split_tagging_fn(df_tag, OneEntryArgs) # the core part. 
        ##########################################################
        Split_to_Selection = OneEntryArgs['Split_Part']['Split_to_Selection']

        split_to_dataset = {}
        for split_name, Selection in Split_to_Selection.items():
            # split_to_dataset[split_name] = dataset.filter(lambda x: apply_multiple_conditions(x, split_config['Rules'], split_config['Op']))
            Rules = Selection['Rules']
            Op = Selection['Op']
        
            index = apply_multiple_conditions(df_tag, Rules, Op)
            indices = np.where(index == 1)[0]
            # len(indices)
            dataset_selected = dataset.select(indices)
            split_to_dataset[split_name] = dataset_selected

        split_to_dataset = datasets.DatasetDict(split_to_dataset)

        # if config is not None:
        #     # config = self.config
        #     # config['CF_to_CFvocab'] = CF_to_CFvocab
        #     dataset_info = DatasetInfo.from_dict({'config_name': config})
        #     split_to_dataset.info.update(dataset_info)

        return split_to_dataset


    def setup_EntryFn_to_NameToData(self, split_to_dataset, CF_to_CFvocab = None, OneEntryArgs = None):
        Name_to_Data = {}
        for split_name, dataset in split_to_dataset.items():
            Name_to_Data[split_name] = {'ds_case': dataset}
        
        if CF_to_CFvocab is None:
            CF_to_CFvocab = self.CF_to_CFvocab
        if OneEntryArgs is None:
            OneEntryArgs = self.OneEntryArgs
        for name, Data in Name_to_Data.items():
            # ds = Data['ds_case']
            s = datetime.now()
            Data = self.setup_EntryFn_to_Data(Data, CF_to_CFvocab, OneEntryArgs)
            # Data['ds_case'] = ds # should not use this.
            e = datetime.now()
            logger.info(f'Name: {name} is transformed in {e-s}')
            Name_to_Data[name] = Data
        return Name_to_Data
    

    