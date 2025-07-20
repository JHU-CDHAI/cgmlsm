import os
import logging
import datasets
import shutil
import pandas as pd
from functools import reduce
from pprint import pprint 
from datetime import datetime

from ..base import Base
logger = logging.getLogger(__name__)

RECORD_FN_PATH = 'fn/fn_record/record'


class RecordFn(Base):
    def __init__(self, RecordName, SPACE):
        self.SPACE = SPACE
        pypath = os.path.join(self.SPACE['CODE_FN'], RECORD_FN_PATH, RecordName + '.py')
        self.pypath = pypath
        self.load_pypath()

    def load_pypath(self):
        self.dynamic_fn_names = ['get_RawRecProc_for_HumanGroup']

        module = self.load_module_variables(self.pypath)
        self.OneRecord_Args = module.OneRecord_Args
        self.RawName_to_RawConfig = module.RawName_to_RawConfig
        self.attr_cols = module.attr_cols
        self.get_RawRecProc_for_HumanGroup = module.get_RawRecProc_for_HumanGroup


class Record(Base):

    def __init__(self, RecordName, human, rec_fn = None):
        self.RecordName = RecordName
        self.human = human
        self.SPACE = human.SPACE
        pypath = os.path.join(self.SPACE['CODE_FN'], RECORD_FN_PATH, RecordName + '.py')
        self.pypath = pypath
        self.rec_fn = rec_fn
        self.record_prt = None
        self.datapath = os.path.join(human.datapath, RecordName)
        self.ds_RecAttr = None 

    def __repr__(self):
        return f'<Record: {self.RecordName}>'

    def setup_fn(self, rec_fn = None):
        if rec_fn is None and self.rec_fn is None:
            rec_fn = RecordFn(self.RecordName, self.SPACE)
        if rec_fn is None and self.rec_fn is not None:
            rec_fn = self.rec_fn
        self.rec_fn = rec_fn

        self.OneRecord_Args = rec_fn.OneRecord_Args
        self.RawName_to_RawConfig = rec_fn.RawName_to_RawConfig
        self.attr_cols = rec_fn.attr_cols
        self.get_RawRecProc_for_HumanGroup = rec_fn.get_RawRecProc_for_HumanGroup
        self.dynamic_fn_names = rec_fn.dynamic_fn_names

    def setup_prt(self, record_prt = None, load_data = True, save_data = True, via_method = 'ds'):
        if record_prt is None:
            OneRecord_Args = self.OneRecord_Args
            human = self.human
            if OneRecord_Args['ParentRecName'] is not None:
                record_prt = Record(OneRecord_Args['ParentRecName'], human)
                record_prt.setup_fn()
                record_prt.setup_prt(load_data = load_data, save_data = save_data, via_method = via_method)
                record_prt.initialize_record(load_data = load_data, save_data = save_data, via_method = via_method)
                self.record_prt = record_prt
            else:
                self.record_prt = None
        else:
            self.record_prt = record_prt


        
    def update_UserTimezone_to_dfHuman(self, human, record, via_method = 'ds'):
        df_Human = human.df_Human
        HumanID = human.OneHuman_Args['HumanID']

        UseTzColName = record.OneRecord_Args['UseTzColName']
        # if UseTzColName is not None: 
        columns = [HumanID, UseTzColName]
        
        # if 'user_tz' not in df_Human.columns and record is not None:
        if hasattr(record, 'ds_RecAttr'):
            ds_RecAttr = record.ds_RecAttr
            df_HumanTimezone = ds_RecAttr.select_columns(columns).to_pandas()
        else:
            df_RecAttr = record.df_RecAttr
            df_HumanTimezone = df_RecAttr[columns]

        df_HumanTimezone = df_HumanTimezone.rename(columns = {UseTzColName: 'user_tz'})
        df_Human = pd.merge(df_Human, df_HumanTimezone, how = 'left')
        human.df_Human = df_Human   # update df_Human
        logger.info(f'update UserTimezone to df_Human: {df_Human.shape} ...')


    def get_RecordPrtInfo(self, 
                          OneRecord_Args = None, 
                          human = None,    
                          record_prt = None,
                          via_method = 'ds'):
        '''
        consider whether we do not need to calculate the RecordPrtInfo again in RecordBase settings. 
        '''
        
        OneRecord_Args = self.OneRecord_Args if OneRecord_Args is None else OneRecord_Args
        human = self.human if human is None else human
        
        OneHuman_Args = human.OneHuman_Args 
        HumanID = OneHuman_Args['HumanID']
        RawHumanID = OneHuman_Args['RawHumanID']
        df_Human = human.df_Human
            
        if record_prt is None:
            # this part is ususally for dfHuman
            PrtRecord_Args = {}
            PrtRecord_Args['RecID'] = HumanID
            PrtRecord_Args['RawRecID'] = RawHumanID
            PrtRecord_Args['RecIDChain'] = [HumanID, RawHumanID]
            df_Prt = df_Human[[HumanID, RawHumanID]]

        elif record_prt.RecordName == human.HumanName:
            # this part is ususally dfHumanRecord
            PrtRecord_Args = {}
            PrtRecord_Args['RecID'] = HumanID
            PrtRecord_Args['RawRecID'] = RawHumanID
            PrtRecord_Args['RecIDChain'] = [HumanID, RawHumanID]
            df_Prt = df_Human[[HumanID, RawHumanID]]

            if 'user_tz' not in df_Human.columns \
                and 'UseTzColName' in record_prt.OneRecord_Args:
                self.update_UserTimezone_to_dfHuman(human, record_prt, via_method)
                logger.info(f'update UserTimezone to df_Human: {df_Human.shape} ...')
                df_Human = human.df_Human # updated df_Human with user_tz

        else:
            PrtRecord_Args = record_prt.OneRecord_Args
            ds_RecAttr_Prt = record_prt.ds_RecAttr
            columns = PrtRecord_Args['RecIDChain']
            RawRecID = PrtRecord_Args['RawRecID']
            RawRecID_List = [RawRecID] if type(RawRecID) == str else RawRecID
            columns = columns + [i for i in RawRecID_List if i not in columns]
            df_Prt = ds_RecAttr_Prt.select_columns(columns).to_pandas()

            
            
        RecordPrtInfo = {
            'PrtRecord_Args': PrtRecord_Args,
            'df_Prt': df_Prt,
        }
        return RecordPrtInfo

    
    def display_Record_RawNameCols(self, RawNameList, RawName_to_dfRaw):

        RawName_to_Sample = {}
        for RawName in RawNameList:
            if RawName not in RawName_to_dfRaw:
                print(f'{RawName} not in RawName_to_dfRaw')
                continue
            dfInfo = RawName_to_dfRaw[RawName]

            print(f'\n========== {RawName} =============')
            if type(dfInfo) == pd.DataFrame:
                dfRaw = dfInfo
                print('---> dfRaw:', dfRaw.shape)
                df = dfRaw.head(5)
                raw_tables_columns = list(df.columns)
            else:
                dfRawPath = dfInfo
                print('---> dfRawPath:', dfRawPath)
                if dfRawPath.endswith('.csv'):
                    df = pd.read_csv(dfRawPath, nrows=5)
                    raw_tables_columns = list(df.columns)
                
                elif dfRawPath.endswith('.parquet'):
                    df = pd.read_parquet(dfRawPath)
                    raw_tables_columns = list(df.columns)

                elif dfRawPath.endswith('.csv.gz'):
                    df = pd.read_csv(dfRawPath, nrows=5, compression='gzip')
                    raw_tables_columns = list(df.columns)
                    
                elif dfRawPath.endswith('.p'):
                    df = pd.read_pickle(dfRawPath)
                    raw_tables_columns = list(df.columns)
            # for i in raw_tables_columns: print('-', i)
            print('----------- copy below -----------------')
            print(f'RawName = "{RawName}"')
            print('raw_columns = ', end='')
            pprint(raw_tables_columns, compact=True)
            print('----------- copy above -----------------')
            
            print('=======================\n\n')
            RawName_to_Sample[RawName] = df
        return RawName_to_Sample
    

    def get_dfHumanSelected(self,
                            OneCohort_Args,
                            OneHuman_Args,
                            OneRecord_Args,
                            df_Human):
        cohort_label = OneCohort_Args['CohortLabel']
        HumanID = OneHuman_Args['HumanID']
        RawHumanID = OneHuman_Args['RawHumanID']
        human_group_size = OneRecord_Args.get('human_group_size', 10000)
        RawNameList = OneRecord_Args['RawNameList']
        dfx = df_Human[[HumanID, RawHumanID, 'CohortLabel'] + RawNameList]
        # dfx = dfx[dfx['CohortLabel'] == cohort_label].reset_index(drop = True)
        dfx = dfx[(dfx[RawNameList] > 0).mean(axis = 1) == 1].reset_index(drop = True)
        dfx['human_group'] = (dfx.index / human_group_size).astype(int)
        df_HumanSelected = dfx
        return df_HumanSelected
    

    def get_dfRawRec_from_dfHumanGroup(self, 
                                       OneHuman_Args, 
                                       df_HumanGroup, 
                                       RawName_to_RawConfig, 
                                       RawName_to_dfRaw):
        RawHumanID = OneHuman_Args['RawHumanID']

        L = []
        for RawName, RawConfig in RawName_to_RawConfig.items():
            raw_columns = RawConfig['raw_columns']
            rec_chunk_size = RawConfig.get('rec_chunk_size', 100000)
            RawHumanID_to_RawNum = dict(zip(df_HumanGroup[RawHumanID], df_HumanGroup[RawName]))
            
            dfRawInfo = RawName_to_dfRaw[RawName]
            if type(dfRawInfo) == str:
                dfRawPath = dfRawInfo
                assert os.path.exists(dfRawPath), f'{dfRawPath} not exists'
                file_size = os.path.getsize(dfRawPath)
                assert file_size > 0, f'{dfRawPath} is empty'
                logger.info(f'RawName "{RawName}" from dfRawPath: {dfRawPath}')
                # Check the file extension
                file_extension = os.path.splitext(dfRawPath)[1].lower()
                if file_extension not in ['.csv', '.csv.gz']:
                    if file_extension == '.parquet':
                        dfRaw = pd.read_parquet(dfRawPath)
                    else:
                        raise ValueError(f'Unsupported file format: {file_extension}')
                else:
                    dfRaw = None

            else:
                dfRawPath = None
                dfRaw = dfRawInfo

            df_HumanRawRec_of_RawTable = self.load_dfRawRec_from_RawHumanIDToRawNum(RawHumanID_to_RawNum, 
                                                                                    RawName, 
                                                                                    RawHumanID, 
                                                                                    dfRawPath, 
                                                                                    dfRaw,
                                                                                    raw_columns, 
                                                                                    rec_chunk_size)
            L.append(df_HumanRawRec_of_RawTable)

        df_RawRec_for_HumanGroup = reduce(lambda left, right: pd.merge(left, right, on= RawHumanID, how='outer'), L)
        return df_RawRec_for_HumanGroup



    def load_dfRawRec_from_RawHumanIDToRawNum(self, 
                                              RawHumanID_to_RawNum, 
                                              RawName, 
                                              RawHumanID, 
                                              dfRawPath, 
                                              dfRaw,
                                              raw_columns, 
                                              rec_chunk_size):

        RawHumanIDList = [i for i in RawHumanID_to_RawNum]


        if type(dfRaw) == pd.DataFrame:
            df_RawRec = dfRaw
            df_RawRec = df_RawRec[df_RawRec[RawHumanID].isin(RawHumanIDList)].reset_index(drop = True)
        else:
            Total_RecNum = sum([v for k, v in RawHumanID_to_RawNum.items()])
            df_RawRec = pd.DataFrame()

            for chunk in pd.read_csv(dfRawPath, chunksize = rec_chunk_size, low_memory = False):
                chunk = chunk[chunk[RawHumanID].isin(RawHumanIDList)]
                if len(chunk) == 0: continue
                df_RawRec = pd.concat([df_RawRec, chunk])
                if len(df_RawRec) == Total_RecNum: break
                if len(df_RawRec) > Total_RecNum: raise ValueError(f'{RawName} more than given number {Total_RecNum}')
            df_RawRec = df_RawRec.reset_index(drop = True)

        if type(raw_columns) == list: 
            for col in raw_columns:
                if col not in df_RawRec.columns:
                    logger.warning(f'{col} not in the df_RawRec data for Record Name <{RawName}>')
                    df_RawRec[col] = None
            df_RawRec = df_RawRec[raw_columns].reset_index(drop = True)
        
        return df_RawRec
    

    def get_dfRecAttr_from_dfRawRec(self, df_RawRecProc, OneRecord_Args, RecordPrtInfo):
        # ---------------------------
        df = df_RawRecProc
        # RecordPrtInfo = OneRecord_Args['RecordPrtInfo']

        # x. merge the df_Prt to the df
        df_Prt = RecordPrtInfo['df_Prt']
        PrtRecord_Args = RecordPrtInfo['PrtRecord_Args']
        df = pd.merge(df_Prt, df, how = 'inner', on = PrtRecord_Args['RawRecID'])

        # y. sort the table by Parent IDs and DT
        RecDT = OneRecord_Args.get('RecDT', None)
        RecID_Chain = PrtRecord_Args['RecIDChain']
        Sorted_Cols = RecID_Chain if RecDT is None else RecID_Chain + [RecDT]
        if len(Sorted_Cols) > 0:
            df = df.sort_values(Sorted_Cols).reset_index(drop = True)

        # z. create a new column for RecID
        RecID = OneRecord_Args['RecID']
        PrtRecID = PrtRecord_Args['RecID']
        if RecID not in df.columns:
            # when current Record is P, RecID is in df.columns, and do not need to do it. 
            df[RecID] = df[PrtRecID].astype(str) + '-' + df.groupby(PrtRecID).cumcount().astype(str).apply(lambda x: str(x).zfill(7))
        #-------------------

        df_RecAttr = df 
        return df_RecAttr


    def get_dsRecAttr(self, 
                     OneRecord_Args = None, 
                     human = None, 
                     RawName_to_RawConfig = None,
                     attr_cols = None,
                     get_RawRecProc_for_HumanGroup = None, 
                     record_prt = None, RecordPrtInfo = None): 
        
        
        if OneRecord_Args is None:
            OneRecord_Args = self.OneRecord_Args

        if human is None:
            human = self.human

        if RawName_to_RawConfig is None:
            RawName_to_RawConfig = self.RawName_to_RawConfig 

        if attr_cols is None:
            attr_cols = self.attr_cols

        if get_RawRecProc_for_HumanGroup is None:
            get_RawRecProc_for_HumanGroup = self.get_RawRecProc_for_HumanGroup

        if record_prt is None:
            record_prt = self.record_prt

        if RecordPrtInfo is None:
            # assert record_prt is not None
            # record_prt could be None for Human-Record. 
            RecordPrtInfo = self.get_RecordPrtInfo(OneRecord_Args, human, record_prt)
        
        RecordName = OneRecord_Args['RecordName']
        OneCohort_Args = human.cohort.OneCohort_Args
        OneHuman_Args = human.OneHuman_Args
        # df_Human = human.df_Human
        RawName_to_dfRaw = human.cohort.RawName_to_dfRaw
        
        
        df_HumanSelected = self.get_dfHumanSelected(OneCohort_Args, OneHuman_Args, OneRecord_Args, human.df_Human)
        

        ds_RecAttr_list = []
        for human_group, df_HumanGroup in df_HumanSelected.groupby('human_group'):
            logger.info(f'=====For Record {self.RecordName}, Human by Batch Group: <{human_group}>=====')
            # display(df_HumanGroup)
            df_RawRec_for_HumanGroup = self.get_dfRawRec_from_dfHumanGroup(OneHuman_Args,
                                                                            df_HumanGroup,
                                                                            RawName_to_RawConfig, 
                                                                            RawName_to_dfRaw)
            logger.info(f'pulled df_RawRec_for_HumanGroup: {df_RawRec_for_HumanGroup.shape}')
            

            logger.info('processing get_RawRecProc_for_HumanGroup ...')
            df_RawRecProc_for_HumanGroup = get_RawRecProc_for_HumanGroup(df_RawRec_for_HumanGroup, OneRecord_Args, human.df_Human)

            logger.info('processing df_RawRecProc_for_HumanGroup ...')
            df_RecAttr_for_HumanGroup = self.get_dfRecAttr_from_dfRawRec(df_RawRecProc_for_HumanGroup, OneRecord_Args, RecordPrtInfo)
            df_RecAttr_for_HumanGroup = df_RecAttr_for_HumanGroup[attr_cols].reset_index(drop=True)

            logger.info(f'current df_RecAttr_for_HumanGroup: {df_RecAttr_for_HumanGroup.shape} ...')
            # ---------------------------------------

            # if len(df_RecAttr_for_HumanGroup) == 0: continue 
            # print(df_RecAttr_for_HumanGroup.columns)
            ds_RecAttr_for_HumanGroup = datasets.Dataset.from_pandas(df_RecAttr_for_HumanGroup)
            ds_RecAttr_list.append(ds_RecAttr_for_HumanGroup)
            del ds_RecAttr_for_HumanGroup

        if len(ds_RecAttr_list) == 0:
            ds_RecAttr = None # <--- for via_method ds, ds_RecAttr could be None
            logger.info(f'current ds_RecAttr for RecName {RecordName}: is empty ...')
        else:
            ds_RecAttr = datasets.concatenate_datasets(ds_RecAttr_list)
            logger.info(f'current ds_RecAttr for RecName {RecordName}: {len(ds_RecAttr)} ...')
        return ds_RecAttr


    def get_dfRecAttr(self, 
                     OneRecord_Args = None, 
                     human = None, 
                     RawName_to_RawConfig = None,
                     attr_cols = None,
                     get_RawRecProc_for_HumanGroup = None, 
                     record_prt = None, RecordPrtInfo = None): 
        
        
        if OneRecord_Args is None:
            OneRecord_Args = self.OneRecord_Args

        if human is None:
            human = self.human

        if RawName_to_RawConfig is None:
            RawName_to_RawConfig = self.RawName_to_RawConfig 

        if attr_cols is None:
            attr_cols = self.attr_cols

        if get_RawRecProc_for_HumanGroup is None:
            get_RawRecProc_for_HumanGroup = self.get_RawRecProc_for_HumanGroup

        if record_prt is None:
            record_prt = self.record_prt

        if RecordPrtInfo is None:
            # assert record_prt is not None
            # record_prt could be None for Human-Record. 
            RecordPrtInfo = self.get_RecordPrtInfo(OneRecord_Args, human, record_prt, via_method = 'df')
        
        RecordName = OneRecord_Args['RecordName']
        OneCohort_Args = human.cohort.OneCohort_Args
        OneHuman_Args = human.OneHuman_Args
        # df_Human = human.df_Human
        RawName_to_dfRaw = human.cohort.RawName_to_dfRaw
        
        # df_HumanSelected = self.get_dfHumanSelected(OneCohort_Args, OneHuman_Args, OneRecord_Args, human.df_Human)
        
        df_HumanGroup = human.df_Human
        df_RawRec_for_HumanGroup = self.get_dfRawRec_from_dfHumanGroup(OneHuman_Args,
                                                                        df_HumanGroup,
                                                                        RawName_to_RawConfig, 
                                                                        RawName_to_dfRaw)
                                                                        
        df_RawRecProc_for_HumanGroup = get_RawRecProc_for_HumanGroup(df_RawRec_for_HumanGroup, OneRecord_Args, human.df_Human)

        df_RecAttr_for_HumanGroup = self.get_dfRecAttr_from_dfRawRec(df_RawRecProc_for_HumanGroup, OneRecord_Args, RecordPrtInfo)
        df_RecAttr_for_HumanGroup = df_RecAttr_for_HumanGroup[attr_cols].reset_index(drop=True)

        logger.info(f'current df_RecAttr_for_HumanGroup: {df_RecAttr_for_HumanGroup.shape} ...')
        # ---------------------------------------

        df_RecAttr = df_RecAttr_for_HumanGroup

        logger.info(f'current ds_RecAttr for RecName {RecordName}: {len(df_RecAttr)} ...')
        
        # assert len(df_RecAttr) > 0

        if len(df_RecAttr) == 0:
            # df_RecAttr = None
            logger.info(f'current ds_RecAttr for RecName {RecordName}: is empty ...')
        
        return df_RecAttr


    def get_dsRecIndex(self, 
                       OneHuman_Args = None, 
                       OneRecord_Args = None, 
                       ds_RecAttr = None):
        if OneHuman_Args is None:
            OneHuman_Args = self.human.OneHuman_Args
        if OneRecord_Args is None:
            OneRecord_Args = self.OneRecord_Args
        if ds_RecAttr is None:
            ds_RecAttr = self.ds_RecAttr

        if ds_RecAttr is None: 
            ds_Recindex = None # <--- for via_method ds, ds_Recindex could be None
        else:
            RecDT = OneRecord_Args.get('RecDT', None) # ['RecDT']
            HumanID = OneHuman_Args['HumanID'] 
            if RecDT and RecDT in ds_RecAttr.column_names:
                df_rec = ds_RecAttr.select_columns([HumanID, RecDT]).to_pandas()
            else:
                df_rec = ds_RecAttr.select_columns([HumanID]).to_pandas()

            s1 = datetime.now()
            RecIndex_generator = RecIndex_Generator(df_rec, HumanID, RecDT)
            ds_Recindex = datasets.Dataset.from_generator(RecIndex_generator)
            e1 = datetime.now()
            logger.info(f'[xxxxxxxxxxxxxxxx] {e1-s1} record.ds_Recindex')

            del df_rec 
        return ds_Recindex


    def get_dfRecIndex(self, 
                       OneHuman_Args = None, 
                       OneRecord_Args = None, 
                       df_RecAttr = None):
        if OneHuman_Args is None:
            OneHuman_Args = self.human.OneHuman_Args
        if OneRecord_Args is None:
            OneRecord_Args = self.OneRecord_Args
        if df_RecAttr is None:
            df_RecAttr = self.df_RecAttr


        # if df_RecAttr is None: 
        #     df_Recindex = None
        #     return df_Recindex

        RecDT = OneRecord_Args.get('RecDT', None) # ['RecDT']
        HumanID = OneHuman_Args['HumanID'] 

        s1 = datetime.now()
        if RecDT and RecDT in df_RecAttr.columns:
            df_rec = df_RecAttr[[HumanID, RecDT]]# .to_pandas()
        else:
            df_rec = df_RecAttr[[HumanID]]# .to_pandas()
            # RecIndex_generator = RecIndex_Generator(df_rec, HumanID, RecDT)
            # ds_Recindex = datasets.Dataset.from_generator(RecIndex_generator)

        L = []
        for HumanIDValue, df_rec_ind in df_rec.groupby(HumanID):
            d = {}
            # d[f'{self.RootID}_idx'] = idx  
            d[HumanID] = HumanIDValue
            d['interval'] = min(df_rec_ind.index), max(df_rec_ind.index)
            if RecDT is not None:
                d['dates'] = [i.isoformat() for i in df_rec_ind[RecDT].tolist()]
            L.append(d)

        df_Recindex = pd.DataFrame(L, columns = [HumanID, 'interval', 'dates'])

        e1 = datetime.now()
        logger.info(f'[xxxxxxxxxxxxxxxx] {e1-s1} record.ds_Recindex')

        # del df_rec 
        return df_Recindex
    

    def load_data(self, datapath = None):
        if datapath is None:
            datapath = self.datapath

        RecName = self.OneRecord_Args['RecordName']
        datapath_RecAttr = os.path.join(datapath, f'RecAttr_{RecName}')
        datapath_RecIndex = os.path.join(datapath, f'RecIndex_{RecName}')

        if os.path.exists(datapath_RecAttr):
            assert os.path.exists(datapath_RecIndex)
            # df_Human = pd.read_csv(datapath_human)
            ds_RecAttr = datasets.load_from_disk(datapath_RecAttr)
            ds_RecIndex = datasets.load_from_disk(datapath_RecIndex)    
            # df_Human = ds_Human.to_pandas()
            # del ds_Human
            logger.info(f"Success to load: {datapath_RecAttr}")
            results = {'ds_RecAttr': ds_RecAttr, 'ds_RecIndex': ds_RecIndex}
        else:
            logger.info(f"Fail to load: record data does not exist: {datapath}")
            results = None 

        return results
    

    def save_data(self, ds_RecAttr = None, ds_RecIndex = None, datapath = None):
        if ds_RecAttr is None:
            ds_RecAttr = self.ds_RecAttr

        if ds_RecIndex is None:
            ds_RecIndex = self.ds_RecIndex

        if datapath is None:
            datapath = self.datapath

        RecName = self.OneRecord_Args['RecordName']
        datapath_RecAttr = os.path.join(datapath, f'RecAttr_{RecName}')
        datapath_RecIndex = os.path.join(datapath, f'RecIndex_{RecName}')

        if self.ds_RecAttr is not None:
            if not os.path.exists(datapath_RecAttr):
                os.makedirs(datapath_RecAttr)

            if not os.path.exists(datapath_RecIndex):
                os.makedirs(datapath_RecIndex)
                
            self.ds_RecAttr.save_to_disk(datapath_RecAttr)
            self.ds_RecIndex.save_to_disk(datapath_RecIndex)
            logger.info(f"Success to save: {datapath_RecAttr}")
        # return {'datapath_RecAttr': datapath_RecAttr, 'datapath_RecIndex': datapath_RecIndex}
    
    def initialize_record(self, load_data = True, save_data = True, via_method = 'ds', shadow_df = True):

        self.shadow_df = shadow_df
        OneHuman_Args = self.human.OneHuman_Args
        # OneRecord_Args = self.OneRecord_Args
        # RecDT = OneRecord_Args.get('RecDT', None) # ['RecDT']
        HumanID = OneHuman_Args['HumanID'] 


        results = None
        # print(load_data, save_data, 'in initialize_record')
        if load_data == True:
            results = self.load_data()

        if results is not None:
            # print(via_method, '----- via method is', via_method)
            self.ds_RecAttr = results['ds_RecAttr']
            self.ds_RecIndex = results['ds_RecIndex']

        else:
            if via_method == 'ds':
                s1 = datetime.now()
                self.ds_RecAttr = self.get_dsRecAttr()
                e1 = datetime.now()
                logger.info(f'xxxxxxxxxxxx {e1-s1} record.get_dsRecAttr')

                s1 = datetime.now()
                self.ds_RecIndex = self.get_dsRecIndex()
                self.df_RecIndex = self.ds_RecIndex.to_pandas().set_index(HumanID)
                e1 = datetime.now()
                logger.info(f'xxxxxxxxxxxx {e1-s1} record.get_dsRecIndex')

                if save_data == True:
                    self.save_data(self.ds_RecAttr, self.ds_RecIndex)

            elif via_method == 'df':
                s1 = datetime.now()
                self.df_RecAttr = self.get_dfRecAttr()
                e1 = datetime.now()
                logger.info(f'xxxxxxxxxxxx {e1-s1} record.get_dfRecAttr')

                s1 = datetime.now()
                self.df_RecIndex = self.get_dfRecIndex()
                self.df_RecIndex = self.df_RecIndex.set_index(HumanID)
                e1 = datetime.now()
                logger.info(f'xxxxxxxxxxxx {e1-s1} record.get_dfRecIndex')

            else:
                raise ValueError
            
        if shadow_df == True and hasattr(self, 'ds_RecAttr') and hasattr(self, 'ds_RecIndex'):
            self.df_RecAttr = self.ds_RecAttr.to_pandas()
            self.df_RecIndex = self.ds_RecIndex.to_pandas().set_index(HumanID)
            del self.ds_RecAttr, self.ds_RecIndex
            logger.info(f'xxxxxxxxxxxx shadow_df: {self.df_RecAttr.shape} ...')
            logger.info(f'xxxxxxxxxxxx del ds_RecAttr, ds_RecIndex ...')



class RecIndex_Generator:
    def __init__(self, df_rec, HumanID, RecDT):
        self.df_rec = df_rec
        self.HumanID = HumanID
        self.RecDT = RecDT

    def __call__(self):
        HumanID = self.HumanID
        RecDT = self.RecDT
        df_rec = self.df_rec
        for HumanIDValue, df_rec_ind in df_rec.groupby(HumanID):
            d = {}
            # d[f'{self.RootID}_idx'] = idx  
            d[HumanID] = HumanIDValue
            d['interval'] = min(df_rec_ind.index), max(df_rec_ind.index)
            if RecDT is not None:
                d['dates'] = [i.isoformat() for i in df_rec_ind[RecDT].tolist()]
            yield d

