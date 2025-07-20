import os
import logging
import pandas as pd
import json
from ..base import Base

logger = logging.getLogger(__name__)


COHORT_FN_PATH = 'fn/fn_record/cohort'

class CohortFn(Base):
    def __init__(self, Source2CohortMethod, SPACE):
        self.SPACE = SPACE
        pypath = os.path.join(self.SPACE['CODE_FN'], COHORT_FN_PATH, Source2CohortMethod + '.py')
        self.pypath = pypath
        self.load_pypath()


    def load_pypath(self):
        """
        Loads the Python module specified by the pypath attribute and assigns the cohort-specific functions to the instance attributes.

        This method performs the following steps:
        1. Initializes the dynamic function names list with default function names.
        2. Loads the module specified by the pypath attribute using the load_module_variables method.
        3. Assigns the SourceFile_SuffixList from the loaded module to the instance attribute.
        4. Assigns the get_RawName_from_SourceFile function from the loaded module to the instance attribute.
        5. Assigns the process_Source_to_Raw function from the loaded module to the instance attribute.
        6. If the get_InferenceEntry function is defined in the module's MetaDict, assigns it to the instance attribute.

        Attributes:
            dynamic_fn_names (list): A list of dynamic function names associated with the cohort.
            SourceFile_SuffixList (list): A list of suffixes for source files associated with the cohort.
            get_RawName_from_SourceFile (function): A function to get the raw name from a source file.
            process_Source_to_Raw (function): A function to process source data to raw data.
            get_InferenceEntry (function, optional): A function to get the inference entry, if defined in the module.

        Example:
            cohort_fn = CohortFn('WellDocV240629', SPACE)
            cohort_fn.load_pypath()
        """
        # Initialize the list of dynamic function names associated with the cohort
        self.dynamic_fn_names = ['get_RawName_from_SourceFile', 'process_Source_to_Raw']

        # Load the module specified by the pypath attribute
        module = self.load_module_variables(self.pypath)

        # Assign the SourceFile_SuffixList from the loaded module to the instance attribute
        self.SourceFile_SuffixList = module.SourceFile_SuffixList

        # Assign the get_RawName_from_SourceFile function from the loaded module to the instance attribute
        self.get_RawName_from_SourceFile = module.get_RawName_from_SourceFile

        # Assign the process_Source_to_Raw function from the loaded module to the instance attribute
        self.process_Source_to_Raw = module.process_Source_to_Raw

        # If the get_InferenceEntry function is defined in the module's MetaDict, assign it to the instance attribute
        if 'get_InferenceEntry' in module.MetaDict:
            self.get_InferenceEntry = module.get_InferenceEntry



    

class Cohort(Base):

    """
    The Cohort class represents a cohort (a collection of data source) in the project, encapsulating its configuration, data paths, and associated functions.

    Attributes:
        OneCohort_Args (dict): Arguments specific to the cohort, including its name and source-to-cohort mapping.
        CohortName (str): The name of the cohort.
        SPACE (dict): A dictionary containing various paths and configurations used in the project.
        Source2CohortName (str): The name used to map the source data to the cohort.
        pypath (str): The path to the source file for the cohort.
        cohort_fn (function): A function associated with the cohort, used for various operations.
        datapath (str): The path to the folder where the cohort's data is stored.
        Inference_Entry (any): An entry used for inference, if applicable.
        SourceFile_SuffixList (list): A list of suffixes for source files associated with the cohort.
        get_RawName_from_SourceFile (function): A function to get the raw name from a source file.
        process_Source_to_Raw (function): A function to process source data to raw data.
        dynamic_fn_names (list): A list of dynamic function names associated with the cohort.
        
        RawName_to_dfRaw (dict): A dictionary mapping raw data names to their corresponding DataFrame objects.

    Methods:
        __init__(self, OneCohort_Args, SPACE, cohort_fn=None, Inference_Entry=None):
            Initializes the Cohort instance with the provided arguments and sets up paths and configurations.

        setup_fn(self, cohort_fn=None):
            Sets up the cohort function and initializes related attributes.

        update_cohort_args(OneCohort_Args, SPACE):
            Static method to update cohort arguments by replacing placeholders in paths with actual values from SPACE.

    Example:
        OneCohort_Args = {
            'CohortName': 'OhioT1DM',
            'Source2CohortName': 'WellDocV240629',
            'FolderPath': '$DATA_RAW$/WellDoc2023CVSDeRx/'
        }
        SPACE = {
            'CODE_FN': '/path/to/code/functions',
            'DATA_RFT': '/path/to/data/rft',
            'DATA_RAW': '/path/to/data/raw'
        }
        cohort = Cohort(OneCohort_Args, SPACE)
    """


    def __init__(self, OneCohort_Args, SPACE, cohort_fn = None, Inference_Entry = None):
        self.OneCohort_Args = self.update_cohort_args(OneCohort_Args, SPACE)
        self.CohortName = OneCohort_Args['CohortName']
        self.SPACE = SPACE
        self.Source2CohortName = OneCohort_Args['Source2CohortName']
        
        # Load the source file
        pypath = os.path.join(SPACE['CODE_FN'], COHORT_FN_PATH, OneCohort_Args['Source2CohortName'] + '.py')
        self.pypath = pypath   
        self.cohort_fn = cohort_fn

        # datafolder
        # here we use SPACE.get('DATA_RFT', '') instead of SPACE.get('DATA_RFT', '--')
        # datapath is actually the RFT path.
        datapath = os.path.join(SPACE.get('DATA_RFT', ''), self.CohortName)
        # if not os.path.exists(datapath): os.makedirs(datapath)
        self.datapath = datapath
        self.Inference_Entry = Inference_Entry

    def setup_fn(self, cohort_fn = None):
        if cohort_fn is None and self.cohort_fn is None:
            cohort_fn = CohortFn(self.Source2CohortName, self.SPACE)
        if cohort_fn is None and self.cohort_fn is not None:
            cohort_fn = self.cohort_fn
        self.cohort_fn = cohort_fn

        self.SourceFile_SuffixList = cohort_fn.SourceFile_SuffixList
        self.get_RawName_from_SourceFile = cohort_fn.get_RawName_from_SourceFile
        self.process_Source_to_Raw = cohort_fn.process_Source_to_Raw
        self.dynamic_fn_names = cohort_fn.dynamic_fn_names

    

    @staticmethod
    def update_cohort_args(OneCohort_Args, SPACE):
        logger.info(f'OneCohort_Args: {OneCohort_Args}')
        if 'FolderPath' in OneCohort_Args:
            OneCohort_Args['FolderPath'] = OneCohort_Args['FolderPath'].replace('$DATA_RAW$', SPACE['DATA_RAW'])
            # if 'Inference' not in OneCohort_Args['FolderPath']:
            assert os.path.exists(OneCohort_Args['FolderPath']), f"FolderPath does not exist: {OneCohort_Args['FolderPath']}"

        if 'SourcePath' in OneCohort_Args:
            if SPACE['DATA_RAW'] not in OneCohort_Args['SourcePath']: 
                DATA_RAW = SPACE['DATA_RAW']
                FolderPath = OneCohort_Args['FolderPath']
                SourcePath = OneCohort_Args['SourcePath']
                logger.info(f'SPACE - DATA_RAW: {DATA_RAW}')
                logger.info(f'OneCohort_Args - FolderPath: {FolderPath}')
                logger.info(f'OneCohort_Args - SourcePath: {SourcePath}')
                OneCohort_Args['SourcePath'] = os.path.join(FolderPath, SourcePath)
                SourcePath = OneCohort_Args['SourcePath']
                assert os.path.exists(SourcePath), f"SourcePath does not exist: {SourcePath}"
        return OneCohort_Args


    @staticmethod 
    def get_SourceFile_List(FolderPath, SourceFile_SuffixList):
        """
        Retrieve a list of source files from a specified folder and its subfolders.

        This function walks through the directory tree rooted at FolderPath and collects
        all files that match the suffixes specified in SourceFile_SuffixList.

        Args:
            FolderPath (str): The path to the root folder where the search should begin.
            SourceFile_SuffixList (list): A list of file suffixes to filter the source files.

        Returns:
            list: A list of full file paths that match the specified suffixes.
        """
        # Ensure the provided folder path exists
        assert os.path.exists(FolderPath)
    
        # Initialize an empty list to store the source file paths
        SourcFile_List = []
        
        # Walk through the directory tree starting from FolderPath
        folder_list = list(os.walk(FolderPath))
        for folder_info in folder_list:
            
            folder, subfolder, files = folder_info
            # print(folder, subfolder, files)
            # if folder == '__processed__': 
            #     logger.info(f'Skip the __processed__ folder: {folder}')
            #     continue
            
            # Filter files that match the suffixes in SourceFile_SuffixList
            files = [file for file in files if file.endswith(tuple(SourceFile_SuffixList))]
            
            # Create full file paths and add them to the source file list
            fullfile_list = [os.path.join(folder, i) for i in files]
            fullfile_list = [i for i in fullfile_list if '__processed__' not in i]
            SourcFile_List = SourcFile_List + fullfile_list
        
        # Return the list of source file paths
        return SourcFile_List
    

    def save_data(self, RawName_to_dfRaw = None):

        if RawName_to_dfRaw is None:
            RawName_to_dfRaw = self.RawName_to_dfRaw


        datapath = self.datapath

        datapath_file = os.path.join(datapath, 'RawName_to_dfRaw.json')
        
        if not os.path.exists(datapath): os.makedirs(datapath)
        RawName_to_dfRaw_Type = self.get_RawName_to_dfRaw_Type(RawName_to_dfRaw)


        # SPACE = self.SPACE
        if RawName_to_dfRaw_Type == 'File':
            with open(datapath_file, 'w') as f:
                RawName_to_dfRaw_absolute = {k: v.replace(self.SPACE['DATA_RAW'], '$DATA_RAW$') for k, v in RawName_to_dfRaw.items()}

                json.dump(RawName_to_dfRaw_absolute, f, indent = 4)
        else:
            logger.info(f"RawName_to_dfRaw_Type: {RawName_to_dfRaw_Type} will not be saved.")


    def load_data(self, datapath = None):
        if datapath is None:
            datapath = self.datapath

        datapath_file = os.path.join(datapath, 'RawName_to_dfRaw.json')
        # assert os.path.exists(datapath), f"datapath does not exist: {datapath}"
        if os.path.exists(datapath_file): # , f"RawName_to_dfRaw.json does not exist: {datapath}"
            with open(datapath_file, 'r') as f:
                RawName_to_dfRaw_absolute = json.load(f)
                RawName_to_dfRaw = {k: v.replace('$DATA_RAW$', self.SPACE['DATA_RAW']) for k, v in RawName_to_dfRaw_absolute.items()}
        else:
            # when you failed to load.
            logger.info(f"Fail to load: RawName_to_dfRaw.json does not exist: {datapath_file}")
            RawName_to_dfRaw = None
        return RawName_to_dfRaw
        
    def initialize_cohort(self, load_data = True, save_data = True):
        OneCohort_Args = self.OneCohort_Args    
        SPACE = self.SPACE
        SourcePath = OneCohort_Args['SourcePath']
        SourceFile_SuffixList = self.SourceFile_SuffixList
        get_RawName_from_SourceFile = self.get_RawName_from_SourceFile
        process_Source_to_Raw = self.process_Source_to_Raw
        SourceFile_List = self.get_SourceFile_List(SourcePath, SourceFile_SuffixList)

        RawName_to_dfRaw = None 
        if load_data == True:
            # sometimes, the loaded RawName_to_dfRaw is None.
            RawName_to_dfRaw = self.load_data()

        if RawName_to_dfRaw is None:
            Inference_Entry = self.Inference_Entry
            if Inference_Entry is None:
                RawName_to_dfRaw = process_Source_to_Raw(OneCohort_Args, SourceFile_List, get_RawName_from_SourceFile, SPACE)
            else:
                RawName_to_dfRaw = process_Source_to_Raw(OneCohort_Args, Inference_Entry, get_RawName_from_SourceFile, SPACE)
            
            if save_data == True:
                self.save_data(RawName_to_dfRaw)
        
        CohortInfo = {"SourceFile_List": SourceFile_List, "RawName_to_dfRaw": RawName_to_dfRaw}
        self.SourceFile_List = SourceFile_List
        self.RawName_to_dfRaw = RawName_to_dfRaw
        self.CohortInfo = CohortInfo 


    @staticmethod
    def get_RawName_to_dfRaw_Type(RawName_to_dfRaw):
        for RawName, dfRaw in RawName_to_dfRaw.items():
            if isinstance(dfRaw, str):
                assert os.path.exists(dfRaw), f"dfRaw does not exist: {dfRaw}"
                return 'File'
            elif isinstance(dfRaw, pd.DataFrame):
                return 'DataFrame'
            else:
                raise ValueError(f"dfRaw is not a DataFrame or a file path: {dfRaw}")

