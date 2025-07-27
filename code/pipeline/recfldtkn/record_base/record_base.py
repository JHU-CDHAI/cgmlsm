import os
import logging
import datasets
import shutil
import pandas as pd
from functools import reduce
from pprint import pprint 
from datetime import datetime 

from ..base import Base
from .cohort import CohortFn, Cohort 
from .human import HumanFn, Human
from .record import RecordFn, Record
from .recfeat import RecFeatFn, RecFeat
from ..case_base.casefnutils.ro import parse_ROName

logger = logging.getLogger(__name__)
 
 
class OneCohort_Record_Base(Base):
    def __init__(self, 
                 CohortName, 
                 HumanRecordRecfeat_Args, 
                 CohortName_to_OneCohortArgs,
                 
                 SPACE,
                 Inference_Entry = None, 
                 Record_Proc_Config = None,
                 ):
        
        self.SPACE = SPACE
        self.CohortName = CohortName
        self.HumanRecordRecfeat_Args = HumanRecordRecfeat_Args
        # self.CohortName_to_OneCohort_Args = self.get_NameToOneCohortArgs()
        self.CohortName_to_OneCohortArgs = CohortName_to_OneCohortArgs
        # self.Ckpd_to_CkpdObsConfig = Ckpd_to_CkpdObsConfig
        self.Record_Proc_Config = Record_Proc_Config
        self.Inference_Entry = Inference_Entry # the input data in the online inference.
        if Inference_Entry is not None and 'TriggerName_to_dfCaseTrigger' in Inference_Entry:
            self.TriggerName_to_dfCaseTrigger = Inference_Entry['TriggerName_to_dfCaseTrigger']
        else:
            self.TriggerName_to_dfCaseTrigger = {}

    @staticmethod
    def parse_ROName(ROName):
        RONameInfo = parse_ROName(ROName)
        return RONameInfo

    def setup_NameToFn(self):
        Name_to_Fn = {}
        # for CohortName in self.CohortNames:
        
        CohortName = self.CohortName
        OneCohort_Args = self.CohortName_to_OneCohortArgs[CohortName]
        # print(OneCohort_Args)
        Source2CohortMethod = OneCohort_Args['Source2CohortName']
        # print(self.SPACE)
        cohort_fn = CohortFn(Source2CohortMethod, self.SPACE)
        Name_to_Fn['C:' + CohortName] = cohort_fn


        HumanRecordRecfeat_Args = self.HumanRecordRecfeat_Args
        for HumanName in HumanRecordRecfeat_Args:
            human_fn = HumanFn(HumanName, self.SPACE)
            Name_to_Fn['H:' + HumanName] = human_fn

            RecordRecFeat_Args = HumanRecordRecfeat_Args[HumanName]
            for RecordName in RecordRecFeat_Args:
                try: 
                    record_fn = RecordFn(RecordName, self.SPACE)
                    Name_to_Fn['R:' + RecordName] = record_fn
                except:
                    logger.warning(f'RecordFn <{RecordName}>is not available')
                    continue

                RecFeat_Args = RecordRecFeat_Args[RecordName]
                for RecFeatName in RecFeat_Args:
                    try:
                        recfeat_fn = RecFeatFn(RecFeatName, self.SPACE)
                        Name_to_Fn['F:' + RecFeatName] = recfeat_fn
                    except:
                        logger.warning(f'RecFeatFn <{RecFeatName}>is not available')

        self.Name_to_Fn = Name_to_Fn


    def setup_InferenceEntry(self, Inference_Entry = None):
        if Inference_Entry is None:
            Inference_Entry = self.Inference_Entry
        self.Inference_Entry = Inference_Entry

    def initialize_NameToObject(self, Name_to_Fn = None, Record_Proc_Config = None):
        if Name_to_Fn is None:
            Name_to_Fn = self.Name_to_Fn

        Record_Proc_Config = self.Record_Proc_Config
        if Record_Proc_Config is None:
            Record_Proc_Config = {}
        via_method = Record_Proc_Config.get('via_method', 'ds')
        save_data = Record_Proc_Config.get('save_data', True)
        load_data = Record_Proc_Config.get('load_data', True)
        shadow_df = Record_Proc_Config.get('shadow_df', True)

        Inference_Entry = self.Inference_Entry
        # if Inference_Entry is not None:
        # save_data, load_data = False, False; via_method = 'df'

    
        # print(save_data, load_data)
        s = datetime.now()
        CohortName = self.CohortName
        OneCohort_Args = self.CohortName_to_OneCohortArgs[CohortName]
        cohort_fn = Name_to_Fn['C:' + CohortName]
        cohort = Cohort(OneCohort_Args, self.SPACE, cohort_fn, Inference_Entry)
        cohort.setup_fn()
        cohort.initialize_cohort(save_data=save_data, load_data = load_data)
        # cohort.cleanup_fn()
        self.cohort = cohort
        e = datetime.now()
        du = (e - s).total_seconds()
        logger.info(f'****** Time: {du} s for Cohort {CohortName}*******\n')

        HumanRecordRecfeat_Args = self.HumanRecordRecfeat_Args
        Name_to_HRF = {}
        # Human
        for HumanName in HumanRecordRecfeat_Args:
            s = datetime.now()
            human_fn = Name_to_Fn['H:' + HumanName]
            human = Human(HumanName, cohort, human_fn)
            human.setup_fn()
            human.initialize_human(save_data=save_data, load_data = load_data)
            Name_to_HRF[HumanName] = human
            e = datetime.now()
            du = (e - s).total_seconds()
            logger.info(f'****** Time: {du} s for Human {HumanName}*******\n')


            # ######### added 2024-10-13, to update human created dfCaseTrigger #########
            if len(self.TriggerName_to_dfCaseTrigger)> 0:
                HumanID = human.OneHuman_Args['HumanID'] 
                RawHumanID = human.OneHuman_Args['RawHumanID']

                TriggerName_list_to_update = []
                for TriggerName, dfCaseTrigger in self.TriggerName_to_dfCaseTrigger.items():
                    # human.setup_trigger(TriggerName, dfCaseTrigger)
                    if HumanID not in dfCaseTrigger.columns:
                        TriggerName_list_to_update.append(TriggerName)

                for TriggerName in TriggerName_list_to_update:
                    dfCaseTrigger = self.TriggerName_to_dfCaseTrigger[TriggerName]
                    df_Human = human.df_Human
                    before = len(dfCaseTrigger)
                    dfCaseTrigger = pd.merge(dfCaseTrigger, df_Human[[HumanID, RawHumanID]], on = RawHumanID, how = 'inner')
                    after = len(dfCaseTrigger)
                    assert before == after
                    self.TriggerName_to_dfCaseTrigger[TriggerName] = dfCaseTrigger
            # ######### added 2024-10-13, to update human created dfCaseTrigger #########

                    

            # Record
            RecordRecFeat_Args = HumanRecordRecfeat_Args[HumanName]
            for RecordName in RecordRecFeat_Args:

                logger.info(f'****** Start: RecordName {RecordName}*******')
                if 'R:' + RecordName not in Name_to_Fn:
                    logger.warning(f'Name_to_Fn does not contain RecordName <{RecordName}>')
                    RecFeat_Args = RecordRecFeat_Args[RecordName]
                    if len(RecFeat_Args) > 0: 
                        RecFeatName_list = [i for i in RecFeat_Args]
                        logger.warning(f'The following RecFeatName will be ignored: {RecFeatName_list}')
                    continue 

                s = datetime.now()
                record_fn = Name_to_Fn['R:' + RecordName]

                OneRecord_Args = record_fn.OneRecord_Args
                RawNameList = OneRecord_Args['RawNameList']
                if any([RawName not in human.df_Human.columns for RawName in RawNameList]):
                    logger.warning(f'Cohort <{CohortName}> does not contain {RecordName}')
                    
                    RecFeat_Args = RecordRecFeat_Args[RecordName]
                    if len(RecFeat_Args) > 0: 
                        RecFeatName_list = [i for i in RecFeat_Args]
                        logger.warning(f'The following RecFeatName will be ignored: {RecFeatName_list}')
                    continue 

                record = Record(RecordName, human, record_fn)

                # -------- TODO: in the future, this need to be updated -------
                if (HumanName, HumanName) in Name_to_HRF:
                    record_prt = Name_to_HRF[(HumanName, HumanName)]
                else:
                    record_prt = None
                # -------------------------------------------------------------
                s1 = datetime.now()
                record.setup_fn()
                e1 = datetime.now()
                logger.info(f'xxxxxxxxxxxx {e1-s1} record.setup_fn')

                s1 = datetime.now()
                record.setup_prt(record_prt = record_prt, save_data=save_data, load_data = load_data, via_method = via_method)
                e1 = datetime.now()
                logger.info(f'xxxxxxxxxxxx {e1-s1} record.setup_prt')

                s1 = datetime.now()
                record.initialize_record(save_data=save_data, load_data = load_data, via_method = via_method, shadow_df = shadow_df)  
                e1 = datetime.now()
                logger.info(f'xxxxxxxxxxxx {e1-s1} record.initialize_record')

                Name_to_HRF[(HumanName, RecordName)] = record
                e = datetime.now()
                du = (e - s).total_seconds()
                logger.info(f'****** Time: {du} s for Record {RecordName}*******\n')


                # RecFeat
                
                # RecFeat_Args = RecordRecFeat_Args[RecordName]
                # for RecFeatName in RecFeat_Args:

                #     logger.info(f'****** Start: RecordFeat {RecFeatName}*******')

                #     if 'F:' + RecFeatName not in Name_to_Fn:
                #         logger.warning(f'Name_to_Fn does not contain RecFeatName: <{RecFeatName}>')

                #     s = datetime.now()
                #     recfeat_fn = Name_to_Fn['F:' + RecFeatName]
                #     recfeat = RecFeat(RecFeatName, record, recfeat_fn)

                #     s1 = datetime.now()
                #     recfeat.setup_fn()
                #     e1 = datetime.now()
                #     logger.info(f'xxxxxxxxxxxx {e1-s1} recfeat.setup_fn')

                #     s1 = datetime.now()
                #     recfeat.setup_recfeat_tokenizer()
                #     e1 = datetime.now()
                #     logger.info(f'xxxxxxxxxxxx {e1-s1} recfeat.setup_recfeat_tokenizer')


                #     s1 = datetime.now()
                #     recfeat.setup_recfeat_tfm()
                #     e1 = datetime.now()
                #     logger.info(f'xxxxxxxxxxxx {e1-s1} recfeat.setup_recfeat_tfm')

                #     s1 = datetime.now()
                #     recfeat.initialize_recfeat(save_data=save_data, load_data = load_data, via_method = via_method)
                #     e1 = datetime.now()
                #     logger.info(f'xxxxxxxxxxxx {e1-s1} recfeat.initialize_recfeat')

                    
                #     Name_to_HRF[(HumanName, RecordName, RecFeatName)] = recfeat
                #     e = datetime.now()
                #     du = (e - s).total_seconds()
                #     logger.info(f'****** Time: {du} s for RecordFeat {RecFeatName}*******\n')


        self.Name_to_HRF = Name_to_HRF



    # update this function
    def __repr__(self):
        return f'<OneCohort_RecordBase: {self.CohortName}> \n---\n' + str(self.HumanRecordRecfeat_Args) + '\n---'
    

    
class Record_Base(Base):
    def __init__(self, 
                 CohortName_list, 
                 HumanRecordRecfeat_Args, 
                 CohortName_to_OneCohortArgs,
                 SPACE,
                 Inference_Entry = None, 
                 Record_Proc_Config = None,
                 ):
        
        self.SPACE = SPACE

        self.CohortName_list = CohortName_list
        self.CohortName_to_OneCohortArgs = CohortName_to_OneCohortArgs

        self.CohortName_to_OneCohortRecordBase = {}
        for CohortName in CohortName_list:
            onecohort_record_base = OneCohort_Record_Base(CohortName = CohortName, 
                                                          HumanRecordRecfeat_Args = HumanRecordRecfeat_Args,
                                                          CohortName_to_OneCohortArgs = CohortName_to_OneCohortArgs,
                                                          SPACE = SPACE,
                                                          Inference_Entry = Inference_Entry, 
                                                          Record_Proc_Config = Record_Proc_Config,
                                                          )
            
            # save_data = True, load_data = True, via_method = 'ds'
            # save_data = Record_Proc_Config['save_data']
            # load_data = Record_Proc_Config['load_data']
            # via_method = Record_Proc_Config['via_method']
            onecohort_record_base.setup_NameToFn()
            onecohort_record_base.initialize_NameToObject(Record_Proc_Config = Record_Proc_Config)
            self.CohortName_to_OneCohortRecordBase[CohortName] = onecohort_record_base

        self.HumanRecordRecfeat_Args = HumanRecordRecfeat_Args
        self.CohortName_to_OneCohortArgs = CohortName_to_OneCohortArgs
        self.Inference_Entry = Inference_Entry




    