import os
import logging
import datasets
import shutil
import pandas as pd
from functools import reduce
from ..base import Base

logger = logging.getLogger(__name__)

HUMAN_FN_PATH = 'fn/fn_record/human'

class HumanFn(Base):    
    
    def __init__(self, HumanName, SPACE):
        self.SPACE = SPACE  
        pypath = os.path.join(self.SPACE['CODE_FN'], HUMAN_FN_PATH, HumanName + '.py')
        self.pypath = pypath
        self.load_pypath()

    def load_pypath(self):
        self.dynamic_fn_names = ['get_RawHumanID_from_dfRawColumns']

        module = self.load_module_variables(self.pypath)
        OneHuman_Args = module.OneHuman_Args
        # self.HumanID = OneHuman_Args['HumanID']
        # self.RawHumanID = OneHuman_Args['RawHumanID']
        # self.HumanIDLength = OneHuman_Args['HumanIDLength']
        # self.chunk_size = OneHuman_Args.get('chunk_size', 100000)
        self.OneHuman_Args = OneHuman_Args
        self.Excluded_RawNameList = module.Excluded_RawNameList
        self.get_RawHumanID_from_dfRawColumns = module.get_RawHumanID_from_dfRawColumns
        

class Human(Base):
    def __init__(self, HumanName, cohort, human_fn = None):

        self.cohort = cohort
        self.HumanName = HumanName
        SPACE = self.cohort.SPACE
        self.SPACE = cohort.SPACE
        pypath = os.path.join(SPACE['CODE_FN'], HUMAN_FN_PATH, HumanName + '.py')
        self.pypath = pypath

        self.human_fn = human_fn

        self.datapath = os.path.join(cohort.datapath, f'Human{HumanName}')

    def __repr__(self):
        return f'<Human: {self.HumanName}>'
        
    def setup_fn(self, human_fn = None):

        if human_fn is None and self.human_fn is None:
            human_fn = HumanFn(self.HumanName, self.SPACE)
        if human_fn is None and self.human_fn is not None:
            human_fn = self.human_fn
        self.human_fn = human_fn

        OneHuman_Args = human_fn.OneHuman_Args
        self.OneHuman_Args = OneHuman_Args
        self.HumanID = OneHuman_Args['HumanID']
        self.RawHumanID = OneHuman_Args['RawHumanID']
        self.HumanIDLength = OneHuman_Args['HumanIDLength']
        self.chunk_size = OneHuman_Args.get('chunk_size', 100000)

        self.Excluded_RawNameList = human_fn.Excluded_RawNameList
        self.get_RawHumanID_from_dfRawColumns = human_fn.get_RawHumanID_from_dfRawColumns

        self.dynamic_fn_names = human_fn.dynamic_fn_names
        
    def cleanup_fn(self):
        self.human_fn = None
        # self.OneHuman_Args = None
        # self.HumanID = None
        # self.RawHumanID = None
        # self.HumanIDLength = None
        # self.chunk_size = None
        # self.Excluded_RawNameList = None
        self.get_RawHumanID_from_dfRawColumns = None

    def display_dfRaw_with_Columns(self, 
                                   RawName_to_dfRaw = None, 
                                   get_RawHumanID_from_dfRawColumns = None):
        
        if RawName_to_dfRaw is None:
            RawName_to_dfRaw = self.cohort.CohortInfo['RawName_to_dfRaw']

        if get_RawHumanID_from_dfRawColumns is None:
            get_RawHumanID_from_dfRawColumns = self.get_RawHumanID_from_dfRawColumns

        for RawName, info in RawName_to_dfRaw.items():
            print(f'\n======= {RawName} =======')
            
            if type(info) == pd.DataFrame:
                dfRaw = info
                raw_table_columns = dfRaw.columns.tolist()
                print(dfRaw.shape, '<--- dfRaw shape')

            else:
                dfRawPath = info
                print(dfRawPath, '<--- file_path')
                
                if type(dfRawPath) == str and os.stat(dfRawPath).st_size == 0: 
                    print(f'Empty! {dfRawPath} is empty')

                elif dfRawPath.endswith('.csv'):
                    df = pd.read_csv(dfRawPath, nrows=0)
                    raw_table_columns = df.columns.tolist()
                    
                elif dfRawPath.endswith('.csv.gz'):
                    df = pd.read_csv(dfRawPath, compression = 'gzip', nrows=5)
                    raw_table_columns = df.columns.tolist()

                elif dfRawPath.endswith('.p'):
                    df = pd.read_pikcle(dfRawPath)
                    raw_table_columns = df.columns.tolist()

                elif dfRawPath.endswith('.parquet'):
                    df = pd.read_parquet(dfRawPath)
                    raw_table_columns = df.columns.tolist()
                else:
                    raise ValueError(f'file type not supported: {dfRawPath}')
                
            RawHumanID = get_RawHumanID_from_dfRawColumns(raw_table_columns)
            print(raw_table_columns, '<--- raw_table_columns')
            print(RawHumanID, '<--- RawHumanID') # Print the identified key identifier column name.

    

    @staticmethod
    def check_RawNameInfo_type(RawName, info):
        if info is None: 
            logger.info(f"'{RawName}' # is None"); return False

        if type(info) == pd.DataFrame and len(info) == 0: 
            logger.info(f"'{RawName}' # empty dataframe"); return False 

        if type(info) == pd.DataFrame: 
            return {'RawName': RawName, 'dfRaw': info}

        if type(info) == str and not os.path.exists(info):
            logger.info(f"'{info}' # file not exists"); return False
        
        if type(info) == str and os.stat(info).st_size == 0: 
            logger.info(f"'{info}' # emtpy file"); return False 
        
        if type(info) == str and info.split('.')[-1] == 'parquet':
            dfRaw = pd.read_parquet(info)
            return {'RawName': RawName, 'dfRaw': dfRaw, 'fileRaw': info}
        
        if type(info) == str and info.split('.')[-1] == 'p':
            dfRaw = pd.read_pickle(info)
            return {'RawName': RawName, 'dfRaw': dfRaw, 'fileRaw': info}
        
        if type(info) == str and '.csv.gz' in info:
            logger.info(f"'{info}' # use csv.gz file")
            return {'RawName': RawName, 'fileRaw': info, 'fileRaw': info}
        
        if type(info) == str and info.split('.')[-1] == 'csv':
            file_size = os.path.getsize(info)
            if file_size < 1_073_741_824:
                dfRaw = pd.read_csv(info, low_memory=False)
                return {'RawName': RawName, 'dfRaw': dfRaw, 'fileRaw': info}
            else:
                logger.info(f"'{info}' # larger than 1GB")
                return {'RawName': RawName, 'fileRaw': info}
            
        raise ValueError(f'file type not supported: <{type(info)}> {info}')

    @staticmethod
    def read_Human2OneRecNum_by_chunk(RawName,
                                        dfRawInfo, 
                                        RawHumanID, 
                                        chunk_size, 
                                        get_RawHumanID_from_dfRawColumns, 
                                        ):
        
        # step 1: get columns
        if type(dfRawInfo) == pd.DataFrame:
            dfRaw = dfRawInfo
            dfRawPath = None
            columns = dfRaw.columns 
        else:
            dfRawPath = dfRawInfo
            if '.parquet' in dfRawPath:
                dfRaw = pd.read_parquet(dfRawPath)
                dfRawInfo = dfRaw
                columns = dfRaw.columns 
            elif 'csv.gz' in dfRawPath:
                columns = pd.read_csv(dfRawPath, nrows=0, compression='gzip').columns
            else:
                columns = pd.read_csv(dfRawPath, nrows=0).columns

        RawHumanID_selected = get_RawHumanID_from_dfRawColumns(columns)
        logger.info(f'RawHumanID_selected: {RawHumanID_selected}')

        # step 2: read the column value counts, but there are no records
        if RawHumanID_selected is None:
            df_results = pd.DataFrame(columns = [RawHumanID, 'RawName', 'RecNum'])
            return df_results 
        
        # step 3: read the column value counts
        if type(dfRawInfo) == pd.DataFrame:
            df_results = dfRaw[RawHumanID_selected].value_counts()
        else:
            if 'csv.gz' in dfRawPath:
                li = [chunk[RawHumanID_selected].value_counts() 
                    for chunk in pd.read_csv(dfRawPath, 
                                            compression='gzip',
                                            usecols = [RawHumanID_selected], chunksize=chunk_size, low_memory=False)]
            else:
                li = [chunk[RawHumanID_selected].value_counts() 
                    for chunk in pd.read_csv(dfRawPath, usecols = [RawHumanID_selected], chunksize=chunk_size, low_memory=False)]
            df_results = pd.concat(li)
            df_results = df_results.groupby(df_results.index).sum()
            
        df_results = df_results.reset_index().rename(columns = {'count': 'RecNum', RawHumanID_selected: RawHumanID})
        df_results['RawName'] = RawName
        return df_results
    

    def get_df_Human2RawNum_on_RawNameTodfRaw(self, 
                                                cohort = None, 
                                                RawName_to_dfRaw = None, 
                                                OneHuman_Args = None,
                                                get_RawHumanID_from_dfRawColumns = None, 
                                                Excluded_RawNameList = None
                                                ):
        # Part 1: Get the configuration
        cohort = self.cohort if cohort is None else cohort
        OneCohort_Args = cohort.OneCohort_Args
        RawName_to_dfRaw = cohort.CohortInfo['RawName_to_dfRaw'] if RawName_to_dfRaw is None else RawName_to_dfRaw


        OneHuman_Args = self.OneHuman_Args if OneHuman_Args is None else OneHuman_Args
        HumanID = OneHuman_Args['HumanID']
        RawHumanID = OneHuman_Args['RawHumanID']
        HumanIDLength = OneHuman_Args['HumanIDLength']
        chunk_size =  OneHuman_Args.get('chunk_size', 10000)
        get_RawHumanID_from_dfRawColumns = self.get_RawHumanID_from_dfRawColumns if get_RawHumanID_from_dfRawColumns is None else get_RawHumanID_from_dfRawColumns
        Excluded_RawNameList = self.Excluded_RawNameList if Excluded_RawNameList is None else Excluded_RawNameList
        

        # Part 2: Read the records from the Raw Data
        Result_List = []
        for RawName, RawInfo in RawName_to_dfRaw.items():
            RawResults = self.check_RawNameInfo_type(RawName, RawInfo)
            if not RawResults: continue
            df_results = self.read_Human2OneRecNum_by_chunk(RawName, 
                                                            RawInfo,
                                                            RawHumanID, 
                                                            chunk_size, 
                                                            get_RawHumanID_from_dfRawColumns, 
                                                            )
            # logger.info(df_results)
            logger.info(f"{RawName}:'path-{RawResults.get('fileRaw', None)}' # {df_results.shape}")
            Result_List.append(df_results)
        logger.info(f'{len(Result_List)} <---- types of dfRec so far')
        
        # Part 3: Merge the records from the Raw Data
        df_all = pd.concat(Result_List, ignore_index=True)
        df_pivot = df_all.pivot(index=RawHumanID, columns='RawName', values='RecNum').reset_index()

        # logger.info(f"df_pivot columns: {df_pivot.columns}")
        
        # Part 4: Filter the records from the Raw Data
        recname_cols = [i for i in df_pivot.columns if i != RawHumanID]
        # -- only consider the records that is in the included_cols. 
        included_cols = [i for i in recname_cols if i not in Excluded_RawNameList]
        rec_count = df_pivot[included_cols].sum(axis = 1)
        df_Human = df_pivot[rec_count > 0].reset_index(drop = True)


        for RawName in RawName_to_dfRaw:
            if RawName not in df_Human.columns:
                df_Human[RawName] = 0

        df_Human['TotalRecNum'] = df_Human[included_cols].sum(axis = 1)
        logger.info(f"The patient cohort size: {len(df_Human)}")
        
        # --------- filter out the records that have no records
        # df_Human = df_Human[df_Human['TotalRecNum'] > 0]
        df_Human = df_Human.sort_values(RawHumanID).reset_index(drop = True)

        if 'i/n' in OneCohort_Args:
            i_n = OneCohort_Args['i/n']
            i, n = i_n.split('/')
            i, n = int(i), int(n)

            # Convert to 0-based index
            i_zero_based = i - 1

            logger.info(f"Splitting data into {n} partitions and selecting partition {i} (1-based indexing)")

            # Calculate the partition size
            total_records = len(df_Human)
            partition_size = total_records // n

            # Calculate start and end indices for the selected partition
            start_idx = i_zero_based * partition_size
            end_idx = (i_zero_based + 1) * partition_size if i < n else total_records

            # Select the i-th partition
            df_Human = df_Human.iloc[start_idx:end_idx].reset_index(drop=True)
            logger.info(f'Total patient number {total_records}')
            logger.info(f"Selected partition {i} with {len(df_Human)} patients (from index {start_idx} to {end_idx - 1})")

        # Part 5: Assign the HumanID
        CohortLabel = OneCohort_Args['CohortLabel']
        df_Human[HumanID] = range(1, len(df_Human) + 1)
        df_Human[HumanID] = df_Human[HumanID].apply(lambda x: int(str(CohortLabel) + str(x).zfill(HumanIDLength)))
        df_Human['CohortLabel'] = CohortLabel
        cols = ['PID'] + [i for i in df_Human.columns if i not in ['PID']]
        df_Human = df_Human[cols].reset_index(drop = True)
        columns = [i for i in RawName_to_dfRaw if i in df_Human.columns] 
        df_Human[columns] = df_Human[columns].astype(float)
        return df_Human


    def save_data(self, df_Human = None):
        if df_Human is None:
            df_Human = self.df_Human

        datapath = self.datapath 
        datapath_human = os.path.join(datapath, 'Human2RawNum')
        if not os.path.exists(datapath): os.makedirs(datapath)
        
        ds_Human = datasets.Dataset.from_pandas(df_Human)
        ds_Human.save_to_disk(datapath_human)
        

    def load_data(self, datapath = None):
        if datapath is None:
            datapath = self.datapath

        datapath_human = os.path.join(datapath, 'Human2RawNum')

        if os.path.exists(datapath_human):
            # df_Human = pd.read_csv(datapath_human)
            ds_Human = datasets.load_from_disk(datapath_human)
            df_Human = ds_Human.to_pandas()
            if len(df_Human) == 0: 
                df_Human = None
                # del ds_Human
                logger.info(f"Fail to load: {datapath_human}")
            else:
                logger.info(f"Success to load: {datapath_human}")
        else:
            logger.info(f"Fail to load: Human2RawNum does not exist: {datapath}")
            df_Human = None
        return df_Human
    
    def initialize_human(self, load_data = True, save_data = True):
        # self.display_dfRaw_with_Columns()
        df_Human = None 
        if load_data == True:
            df_Human = self.load_data()

        if df_Human is None:
            df_Human = self.get_df_Human2RawNum_on_RawNameTodfRaw()
            if save_data == True:
                self.save_data(df_Human)
        
        self.df_Human = df_Human

    