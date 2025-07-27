import os
import logging
import datasets
import shutil
import pandas as pd
import numpy as np
from functools import reduce
from pprint import pprint 
import itertools
import inspect

from ..base import Base
logger = logging.getLogger(__name__)

from datasets import disable_caching
disable_caching()

RECFEAT_FN_PATH = 'fn/fn_record/recfeat'
RECFEAT_TYPE_LIST = ['Cate', 'N2C', 'Nume', 'External']


dynamic_fn_names = ['RecFeat_Tokenizer_fn', 'get_idx2tkn_fn', 'RecFeat_Tokenizer', 'get_idx2tkn']

class RecFeatFn(Base):
    def __init__(self, RecFeatName, SPACE):
        self.SPACE = SPACE
        self.dynamic_fn_names = dynamic_fn_names
        pypath = os.path.join(self.SPACE['CODE_FN'], RECFEAT_FN_PATH, RecFeatName.replace('-', '_') + '.py')
        self.pypath = pypath
        try:
            self.load_pypath()
        except:
            logger.info(f'No {RecFeatName} found')

    def load_pypath(self):
        # self.dynamic_fn_names = ['get_RawName_from_SourceFile', 'process_Source_to_Raw']
        module = self.load_module_variables(self.pypath)
        self.OneRecFeat_Args = module.OneRecFeat_Args
        self.idx2tkn = module.idx2tkn
        self.RecFeat_Tokenizer_fn = module.RecFeat_Tokenizer_fn
        self.get_idx2tkn_fn = module.get_idx2tkn_fn
        
        OneRecFeat_Args = self.OneRecFeat_Args
        SPACE = self.SPACE
        if 'external_source' in OneRecFeat_Args:
            df_external = self.load_exteranl_data(OneRecFeat_Args, SPACE)
            self.OneRecFeat_Args['df_external'] = df_external
        
    @staticmethod
    def load_exteranl_data(OneRecFeat_Args, SPACE):
        OneRecFeat_Args = OneRecFeat_Args
        external_source = OneRecFeat_Args['external_source']
        external_source = external_source.replace('$DATA_EXTERNAL$', SPACE['DATA_EXTERNAL'])
        # print(external_source)
        df_external = pd.read_pickle(external_source)
        return df_external
        
        
class RecFeatMethod:

    def get_Attr_to_AttrConfig_for_Cate(self, df_RecAttr, OneRecFeat_Args):
        ############################### for Cate Tkn only
        Attr_to_AttrConfig = {}
        Attr_to_AttrMeta = OneRecFeat_Args['Attr_to_AttrMeta']
        TOP_NUM = OneRecFeat_Args.get('TOP_NUM', 30)
        cols = OneRecFeat_Args['value_cols']
        for col in cols:
            AttrMeta = Attr_to_AttrMeta.get(col, {})
            TOP_NUM = AttrMeta.get('TOP_NUM', TOP_NUM)
            top_tkn = list(df_RecAttr[col].value_counts().iloc[:TOP_NUM].index)
            logger.info(f'\n<{col}> Unique: {len(top_tkn)}, top_tkn: {top_tkn}')
            Attr_to_AttrConfig[col] = {'top_tkns': top_tkn} # tolist()
        ###############################
        return Attr_to_AttrConfig
    
    def get_Attr_to_AttrConfig_for_N2C(self, df_RecAttr, OneRecFeat_Args):
        Attr_to_AttrConfig = {}
        cols = OneRecFeat_Args['value_cols']
        for col in cols:
            Attr_to_AttrConfig[col] = {'Max': None, 'Min': None, 'INTERVAL': None}
        return Attr_to_AttrConfig
    
    # ----------------- Get idx2tkn -----------------
    def post_process_recfeat(self, d):
        tkn = list(d.keys())
        wgt = list(d.values())
        output = {'tkn': tkn, 'wgt': wgt}
        return output

    def RecFeat_Tokenizer_for_Cate(self, rec, OneRecFeat_Args):
        d = {}
        Attr_to_AttrConfig = OneRecFeat_Args[f'Attr_to_AttrConfig']
        for attr, AttrConfig in Attr_to_AttrConfig.items():
            top_tkns = AttrConfig['top_tkns']
            value = rec.get(attr, 'unk')
            if value not in top_tkns and value != 'unk': value = 'minor'
            key_value = f"{attr}_{value}"  # Concatenate key and value
            d[key_value] = 1

        output = self.post_process_recfeat(d)   
        return output
      
    def RecFeat_Tokenizer_for_N2C(self, rec, OneRecFeat_Args):
        d = {}
        Attr_to_AttrConfig = OneRecFeat_Args['Attr_to_AttrConfig']
        for attr, AttrConfig in Attr_to_AttrConfig.items():
            Max = AttrConfig['Max']
            Min = AttrConfig['Min']

            Length = len(str(int(Max)))
            
            
            INTERVAL = AttrConfig['INTERVAL']
            if pd.isnull(rec.get(attr, None)) :
                d[f"{attr}:None"] = 1
            elif float(rec[attr]) == Max:
                d[ f"{attr}:Equal{Max}"] = 1
            elif float(rec[attr]) == Min:
                d[ f"{attr}:Equal{Min}"] = 1
            elif float(rec[attr]) > Max:
                d[ f"{attr}:Above{Max}"] = 1
            elif float(rec[attr]) < Min:
                d[ f"{attr}:Below{Min}"] = 1
            else:
                lower_bound = int((float(rec[attr]) // INTERVAL) * INTERVAL)
                upper_bound = int(lower_bound + INTERVAL)

                # Calculate the proportion of value within the interval
                # proportion = (float(rec[attr]) - lower_bound) / INTERVAL
                proportion = round((float(rec[attr]) - lower_bound) / INTERVAL, 4)
                # Construct the keys
                key1 = f"{attr}:{str(lower_bound).zfill(Length)}~{str(upper_bound).zfill(Length)}"
                key2 = f"{key1}Itv"
                # Add them to the dictionary with appropriate weights
                d[key1] = 1
                d[key2] = proportion
        output = self.post_process_recfeat(d)   
        return output
    
    def RecFeat_Tokenizer_for_Nume(self, rec, OneRecFeat_Args):
        d = {}
        for col in OneRecFeat_Args['value_cols']:
            x = rec[col]
            if pd.isnull(x):
                d[f'{col}_None'] = 1
            else:
                d[col] = float(x)
        output = self.post_process_recfeat(d)   
        return output
    
    # ----------------- Get idx2tkn -----------------
    def get_idx2tkn_for_Cate(self, OneRecFeat_Args, RecFeat_Tokenizer):
        ############################################## Cate Tkn only
        assert OneRecFeat_Args['RecFeatType'] == 'Cate'
        Attr_to_AttrConfig = OneRecFeat_Args['Attr_to_AttrConfig']  

        idx2tkn = []
        for attr, AttrConfig in Attr_to_AttrConfig.items():
            top_tkns = AttrConfig['top_tkns']
            idx2tkn = idx2tkn + [f'{attr}_unk', f'{attr}_minor']
            for val in top_tkns:
                idx2tkn.append(f"{attr}_{val}")
        # print(len(idx2tkn))
        # print(idx2tkn[:10])
        ##############################################
        return idx2tkn
    
    def get_idx2tkn_for_N2C(self, OneRecFeat_Args, RecFeat_Tokenizer):
        ############################################## for N2C only
        assert OneRecFeat_Args['RecFeatType'] == 'N2C'
        value_cols = OneRecFeat_Args['value_cols']
        Attr_to_AttrConfig = OneRecFeat_Args[f'Attr_to_AttrConfig']
        # df_simu_list = [pd.DataFrame([{i: None for i in value_cols}])]

        dict_cols = {col: [None] for col in value_cols}
        for col in value_cols:
            AttrConfig = Attr_to_AttrConfig[col]
            Min, Max = AttrConfig['Min'], AttrConfig['Max']
            INTERVAL = AttrConfig['INTERVAL']
            Min_left, Max_right = Min - INTERVAL * 1.5, Max + INTERVAL * 1.5
            values = [Min, Max] + list(np.arange(Min_left, Max_right, INTERVAL))
            dict_cols[col] += values
            for other_col in value_cols:
                if other_col != col: 
                    dict_cols[other_col] += [None] * len(values) 

        df_simu = pd.DataFrame(dict_cols)
        # df_simu.to_csv('tmp.csv')
        df_sim = pd.DataFrame(df_simu.apply(lambda rec: RecFeat_Tokenizer(rec, OneRecFeat_Args), axis = 1).to_list())
        idx2tkn = sorted(list(set(itertools.chain(*df_sim['tkn'].to_list()))))
        ##############################################
        return idx2tkn
    
    def get_idx2tkn_for_Nume(self, OneRecFeat_Args, RecFeat_Tokenizer):
        assert OneRecFeat_Args['RecFeatType'] == 'Nume'
        ############################################## for Nume
        idx2tkn = OneRecFeat_Args['value_cols'] + [f'{col}_None' for col in OneRecFeat_Args['value_cols']]
        ##############################################
        return idx2tkn  
    

class RecFeat(Base, RecFeatMethod):
    def __init__(self, RecFeatName, record, recfeat_fn = None):
        self.RecFeatName = RecFeatName
        self.record = record
        self.SPACE = record.SPACE
        
        pypath = os.path.join(self.SPACE['CODE_FN'], RECFEAT_FN_PATH, RecFeatName.replace('-', '_') + '.py')
        self.pypath = pypath
        self.recfeat_fn = recfeat_fn

        self.datapath = self.record.datapath

        self.dynamic_fn_names = dynamic_fn_names 

    def __repr__(self):
        return f'<RecFeat: {self.RecFeatName}>'


    def get_dfRecAttr(self, record, OneRecFeat_Args, MAX_SAMPLE):
        ds_RecAttr = record.ds_RecAttr

        attr_cols = OneRecFeat_Args['prefix_cols'] + OneRecFeat_Args['value_cols']
        ds_RecAttr = ds_RecAttr.select_columns(attr_cols)
        if len(ds_RecAttr) > MAX_SAMPLE:
            ds_RecAttr = ds_RecAttr.select(range(MAX_SAMPLE))

        df_RecAttr = ds_RecAttr.to_pandas() 
        return df_RecAttr
    

    def display_dfRecAttr(self, df_RecAttr, OneRecFeat_Args):
        if OneRecFeat_Args['RecFeatType'] == 'Cate':
            for col in OneRecFeat_Args['value_cols']:
                logger.info(f'\n----- Describing: <{col}> -----\n{col}: {df_RecAttr[col].value_counts()}\n\n')
        elif OneRecFeat_Args['RecFeatType'] == 'External':
            for col in OneRecFeat_Args['value_cols']:
                logger.info(f'\n----- Describing: <{col}> -----\n{col}: {df_RecAttr[col].value_counts()}\n\n')
        else:
            cols = OneRecFeat_Args['value_cols']
            descp = df_RecAttr[cols].astype(float).describe().round(2)#.to_dict()
            logger.info(f'\n----- Describing: <{cols}> -----\n {descp}')

                
    def get_Attr_to_AttrConfig(self, df_RecAttr, OneRecFeat_Args):
        Attr_to_AttrConfig = {}
        if OneRecFeat_Args['RecFeatType'] == 'Cate':
            Attr_to_AttrConfig = self.get_Attr_to_AttrConfig_for_Cate(df_RecAttr, OneRecFeat_Args)

        elif OneRecFeat_Args['RecFeatType'] == 'N2C':
            print(f'!!! You need to update for N2C AttrConfig yourself...')
            Attr_to_AttrConfig = self.get_Attr_to_AttrConfig_for_N2C(df_RecAttr, OneRecFeat_Args)

        else:
            print(f'You do not need to prepare AttrConfig')
            Attr_to_AttrConfig = None

        return Attr_to_AttrConfig


    def setup_fn(self, recfeat_fn = None):
        if recfeat_fn is None and self.recfeat_fn is None:
            recfeat_fn = RecFeatFn(self.RecFeatName, self.SPACE)
            recfeat_fn.load_pypath()
        if recfeat_fn is None and self.recfeat_fn is not None:
            recfeat_fn = self.recfeat_fn

        self.recfeat_fn = recfeat_fn
        # self.recfeat_fn.load_pypath() 
        self.OneRecFeat_Args = recfeat_fn.OneRecFeat_Args
        self.RecFeat_Tokenizer_fn = recfeat_fn.RecFeat_Tokenizer_fn
        self.get_idx2tkn_fn = recfeat_fn.get_idx2tkn_fn
        self.idx2tkn = recfeat_fn.idx2tkn
        self.dynamic_fn_names = recfeat_fn.dynamic_fn_names
        

    def setup_recfeat_tokenizer(self, OneRecFeat_Args = None, RecFeat_Tokenizer_fn = None):
        if OneRecFeat_Args is None:
            OneRecFeat_Args = self.recfeat_fn.OneRecFeat_Args

        if RecFeat_Tokenizer_fn is None:
            RecFeat_Tokenizer_fn = self.RecFeat_Tokenizer_fn
        
        RecFeatType = OneRecFeat_Args['RecFeatType']
        if self.function_is_empty(RecFeat_Tokenizer_fn):
            if RecFeatType == 'Cate':
                self.RecFeat_Tokenizer = self.RecFeat_Tokenizer_for_Cate
            elif RecFeatType == 'N2C':
                self.RecFeat_Tokenizer = self.RecFeat_Tokenizer_for_N2C
            elif RecFeatType == 'Nume':
                self.RecFeat_Tokenizer = self.RecFeat_Tokenizer_for_Nume
            elif RecFeatType == 'External':
                self.RecFeat_Tokenizer = self.RecFeat_Tokenizer_for_EXT
            else:
                raise ValueError(f'Unknown RecFeatType: {RecFeatType}')
        else:
            self.RecFeat_Tokenizer = RecFeat_Tokenizer_fn


    def setup_recfeat_idx2tkn(self, OneRecFeat_Args = None, get_idx2tkn_fn = None):
        if OneRecFeat_Args is None:
            OneRecFeat_Args = self.recfeat_fn.OneRecFeat_Args
        if get_idx2tkn_fn is None:
            get_idx2tkn_fn = self.get_idx2tkn_fn
        
        RecFeatType = OneRecFeat_Args['RecFeatType']
        if self.function_is_empty(get_idx2tkn_fn):
            if RecFeatType == 'Cate':
                self.get_idx2tkn = self.get_idx2tkn_for_Cate
            elif RecFeatType == 'N2C':
                self.get_idx2tkn = self.get_idx2tkn_for_N2C
            elif RecFeatType == 'Nume':
                self.get_idx2tkn = self.get_idx2tkn_for_Nume
            elif RecFeatType == 'External':
                self.get_idx2tkn = self.get_idx2tkn_for_EXT
            else:
                raise ValueError(f'Unknown RecFeatType: {RecFeatType}')
        else:
            self.get_idx2tkn = get_idx2tkn_fn


    def setup_recfeat_tfm(self, RecFeat_Tokenizer = None, idx2tkn = None, OneRecFeat_Args = None):
        if RecFeat_Tokenizer is None:
            RecFeat_Tokenizer = self.RecFeat_Tokenizer
        if idx2tkn is None:
            idx2tkn = self.idx2tkn
        if OneRecFeat_Args is None:
            OneRecFeat_Args = self.recfeat_fn.OneRecFeat_Args

        recfeat_tfm = RecFeat_Tfm(RecFeat_Tokenizer, idx2tkn, OneRecFeat_Args)
        self.recfeat_tfm = recfeat_tfm
        

    def get_dsRecFeat(self, record = None, OneRecFeat_Args = None, recfeat_tfm = None):
        if record is None:
            record = self.record
        if OneRecFeat_Args is None:
            OneRecFeat_Args = self.OneRecFeat_Args
        if recfeat_tfm is None:
            recfeat_tfm = self.recfeat_tfm

        # df_RecAttr = self.get_dfRecAttr(record, OneRecFeat_Args, MAX_SAMPLE)
        ds_RecAttr = record.ds_RecAttr  
        # num_proc = max(4, OneRecFeat_Args['num_proc'])
        num_proc = OneRecFeat_Args['num_proc']
        batch_size = OneRecFeat_Args['batch_size']


        # for via_method ds: ds_RecFeat could be None.
        if ds_RecAttr is None: 
            ds_RecFeat = None 
        else:
            ds_RecFeat = ds_RecAttr.map(recfeat_tfm, 
                                        batched = True, 
                                        num_proc=num_proc, 
                                        batch_size=batch_size,
                                        load_from_cache_file=False,
                                        )
        

            if 'RecDT' in record.OneRecord_Args:
                columns_to_remove = [i for i in ds_RecAttr.column_names if i not in [record.OneRecord_Args['RecID'], record.OneRecord_Args['RecDT']]]
            else:
                columns_to_remove = ds_RecAttr.column_names # [i for i in ds_RecAttr.column_names if i != record.OneRecord_Args['RecDT']]

            ds_RecFeat = ds_RecFeat.remove_columns(columns_to_remove)

        return ds_RecFeat


    def get_dfRecFeat(self, record = None, OneRecFeat_Args = None, RecFeat_Tokenizer = None):
        if record is None:
            record = self.record
        if OneRecFeat_Args is None:
            OneRecFeat_Args = self.OneRecFeat_Args
        if RecFeat_Tokenizer is None:
            # recfeat_tfm = self.recfeat_tfm
            RecFeat_Tokenizer = self.RecFeat_Tokenizer

        # df_RecAttr = self.get_dfRecAttr(record, OneRecFeat_Args, MAX_SAMPLE)
        df_RecAttr = record.df_RecAttr  # carb 


        ####################
        # assert len(df_RecAttr) > 0
        ####################


        # num_proc = OneRecFeat_Args['num_proc']
        # batch_size = OneRecFeat_Args['batch_size']
        # if len(df_RecAttr) == 0: return None 

        
        df_RecFeat = df_RecAttr.apply(lambda rec: RecFeat_Tokenizer(rec, OneRecFeat_Args), axis = 1)
        df_RecFeat = pd.DataFrame(df_RecFeat.to_list())
        # logger.info(f"df_RecFeat: {df_RecFeat}")
        RecFeatName = self.RecFeatName

        if 'RecDT' in record.OneRecord_Args and record.OneRecord_Args['RecDT'] is not None:
            columns_prefix = [record.OneRecord_Args['RecID'], record.OneRecord_Args['RecDT']]
            df_RecFeat = pd.concat([df_RecAttr[columns_prefix], df_RecFeat], axis = 1)


        if len(df_RecAttr) > 0: 
            # logger.info(df_RecFeat)
            idx2tkn = self.idx2tkn
            tkn2idx = {v:k for k, v in enumerate(idx2tkn)}  
            df_RecFeat['tid'] = df_RecFeat['tkn'].apply(lambda x: [tkn2idx[i] for i in x])

            df_RecFeat[f'{RecFeatName}_tid'] = df_RecFeat['tid'].to_list()
            del df_RecFeat['tid'], df_RecFeat['tkn']
            if 'wgt' in df_RecFeat.columns:
                df_RecFeat[f'{RecFeatName}_wgt'] = df_RecFeat['wgt'].to_list()
                del df_RecFeat['wgt']
        else:

            # prefix_cols = OneRecFeat_Args['prefix_cols']
            # for col in prefix_cols:
            #     df_RecFeat[col] = None

            df_RecFeat[f'{RecFeatName}_tid'] = None 
            use_wgt = OneRecFeat_Args.get('use_wgt', True)
            if use_wgt:
                df_RecFeat[f'{RecFeatName}_wgt'] = None
            # del df_RecFeat['tid'], df_RecFeat['tkn']
            # if 'wgt' in df_RecFeat.columns:
                # del df_RecFeat['wgt']

        # print(df_RecFeat)
        return df_RecFeat
    

    def load_data(self, datapath = None):
        if datapath is None:
            datapath = self.datapath

        RecFeatName = self.OneRecFeat_Args['RecFeatName']
        datapath_RecFeat = os.path.join(datapath, f'RecFeat_{RecFeatName}')
        
        if os.path.exists(datapath_RecFeat):
            # assert os.path.exists(datapath_RecIndex)
            # df_Human = pd.read_csv(datapath_human)
            ds_RecFeat = datasets.load_from_disk(datapath_RecFeat)
            # ds_RecIndex = datasets.load_from_disk(datapath_RecIndex)    
            # df_Human = ds_Human.to_pandas()
            # del ds_Human
            logger.info(f"Success to load: {datapath_RecFeat}")
            results = {'ds_RecFeat': ds_RecFeat}
        else:
            logger.info(f"Fail to load: {RecFeatName} does not exist: {datapath}")
            results = None 

        return results
    
    def save_data(self, ds_RecFeat = None, datapath = None):
        if ds_RecFeat is None:
            ds_RecFeat = self.ds_RecFeat

        if datapath is None:
            datapath = self.datapath

        RecFeatName = self.OneRecFeat_Args['RecFeatName']
        datapath_RecFeat = os.path.join(datapath, f'RecFeat_{RecFeatName}')

        if ds_RecFeat is not None:
            self.ds_RecFeat.save_to_disk(datapath_RecFeat)
            logger.info(f"Success to save: {ds_RecFeat}")
        else:
            logger.info(f"Fail to save: {RecFeatName} does not exist: {datapath}")
        # return {'datapath_RecAttr': datapath_RecAttr, 'datapath_RecIndex': datapath_RecIndex}

    def initialize_recfeat(self, load_data = True, save_data = True, via_method = 'ds'):
        results = None 
        if load_data:
            results = self.load_data()
        
        if results is not None:
            self.ds_RecFeat = results['ds_RecFeat']
        else:

            if via_method == 'ds':
                self.ds_RecFeat = self.get_dsRecFeat()  
                if save_data:
                    self.save_data()
            else:
                self.df_RecFeat = self.get_dfRecFeat()  





class RecFeat_Tfm(Base):
    def __init__(self, RecFeat_Tokenizer, idx2tkn, OneRecFeat_Args):
        self.dynamic_fn_names = dynamic_fn_names
        self.RecFeat_Tokenizer = RecFeat_Tokenizer
        self.idx2tkn = idx2tkn
        self.tkn2idx = {v:k for k, v in enumerate(idx2tkn)}
        self.OneRecFeat_Args = OneRecFeat_Args

    def __call__(self, examples):

        RecFeat_Tokenizer = self.RecFeat_Tokenizer
        OneRecFeat_Args = self.OneRecFeat_Args
        tkn2idx = self.tkn2idx
        RecFeatName = OneRecFeat_Args['RecFeatName']

        df_rec = pd.DataFrame({k:v for k, v in examples.items()})
        output_series = df_rec.apply(lambda x: RecFeat_Tokenizer(x, OneRecFeat_Args), axis=1)

        df_output = pd.DataFrame(output_series.tolist())
        df_output['tid'] = df_output['tkn'].apply(lambda x: [tkn2idx[i] for i in x])   

        cols = ['tid', 'wgt']

        cols = [i for i in cols if i in df_output.columns]
        df_output = df_output[cols]
        
        examples = {}
        for col in cols:
            # examples[f'{RecFeatName}_tid'] = df_output['tid'].to_list()
            # examples[f'{RecFeatName}_wgt'] = df_output['wgt'].to_list()
            examples[f'{RecFeatName}_{col}'] = df_output[col].to_list()

        return examples
