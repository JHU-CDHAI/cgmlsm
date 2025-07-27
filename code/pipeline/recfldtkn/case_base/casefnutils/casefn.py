import os
import logging
import pandas as pd
from pprint import pprint
from ...base import Base
logger = logging.getLogger(__name__)

CASESET_CASEFN_PATH = 'fn/fn_case/case_casefn'



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


    combined_roinfo = [i for i in combined_roinfo]
    return combined_roinfo




class Case_Fn(Base):
    def __init__(self, CaseFnName, SPACE):
        self.SPACE = SPACE
        self.CaseFnName = CaseFnName
        self.pypath = os.path.join(self.SPACE['CODE_FN'], CASESET_CASEFN_PATH, CaseFnName + '.py')
        self.load_pypath()

    def load_pypath(self):
        module = self.load_module_variables(self.pypath)
        self.dynamic_functions = {
            'fn_CaseFn': module.fn_CaseFn,
        }
        self.CaseFnName = module.CaseFnName
        self.Ckpd_to_CkpdObsConfig = module.Ckpd_to_CkpdObsConfig
        self.RO_to_ROName = module.RO_to_ROName
        self.ROName_to_RONameInfo = module.ROName_to_RONameInfo
        self.HumanRecordRecfeat_Args = module.HumanRecordRecfeat_Args
        self.COVocab = module.COVocab

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpickleable functions
        del state['dynamic_functions']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reloading the dynamic functions
        self.load_pypath()

    def get_function(self, func_name):
        return self.dynamic_functions.get(func_name)



def get_CaseFnNameToCaseFnInfo(
        CaseFnName_list, 
        SPACE,
        ):
    CaseFnName_to_CaseFnInfo = {}
    for CaseFnName in CaseFnName_list:
        case_fn = Case_Fn(CaseFnName, SPACE)

        CaseFnInfo = {}
        CaseFnInfo['CaseFnName'] = CaseFnName
        CaseFnInfo['COvocab'] = case_fn.COVocab
        CaseFnInfo['Ckpd_to_CkpdObsConfig'] = case_fn.Ckpd_to_CkpdObsConfig
        CaseFnInfo['RO_to_ROName'] = case_fn.RO_to_ROName
        CaseFnInfo['ROName_list'] = [i for i, v in case_fn.ROName_to_RONameInfo.items()]
        CaseFnInfo['ROName_to_RONameInfo'] = case_fn.ROName_to_RONameInfo
        CaseFnInfo['HumanRecordRecfeat_Args'] = case_fn.HumanRecordRecfeat_Args
        # CaseFnInfo['COVocab'] = case_fn.COVocab
        CaseFnInfo['case_fn'] = case_fn # case_fn.get_function('fn_CaseFn')

        CaseFnName_to_CaseFnInfo[CaseFnName] = CaseFnInfo

    return CaseFnName_to_CaseFnInfo


def get_CaseFnNameToCaseFnData_for_OneCaseExample(case_example, 
                                                  CaseFnName_to_CaseFnInfo,
                                                  ROName_to_ROInfo,
                                                  ROName_to_ROData,
                                                  caseset, # <--- this is weird, what do you want here? the caseset information?
                                                           # <--- maybe remove it in the future. 
                                                  ):
    CaseFnNameField_to_CaseFnData = {}

    # ------------ Calculation Part -----------
    for CaseFnName, CaseFnInfo in CaseFnName_to_CaseFnInfo.items():
        

        # COInfo = COName_to_COInfo[COName]
        
        ROName_list = CaseFnInfo['ROName_list']
        # print('ROName_list:', ROName_list)
        COvocab = CaseFnInfo['COvocab']
        case_fn = CaseFnInfo['case_fn']
        fn_CaseFn = case_fn.get_function('fn_CaseFn')

        CaseData = fn_CaseFn(case_example, 
                            ROName_list,
                            ROName_to_ROData, 
                            ROName_to_ROInfo, 
                            COvocab, 
                            caseset) 
        # -------------------------- fn_CasePhi
        
        # HUMANid, COid = HUMANid_COid
        # CO_to_Cache[HUMANid][COid] = COData

        for k, v in CaseData.items():
            CaseFnNameField_to_CaseFnData[CaseFnName + '-' + k] = v
            
    return CaseFnNameField_to_CaseFnData
            

