import os 
import json 
import logging
import traceback
import pandas as pd 
import copy 
# from ..aidata_base.aidata import AIData 
from ..aidata_base.entry import EntryAIData_Builder

logger = logging.getLogger(__name__)

SPECIAL_FOLDERS = ['pipeline', 'external']

logger = logging.getLogger(__name__)




def load_model_configurations(model_checkpoint_path, SPACE):

    # --------------- load aidata ----------------
    # data_path = os.path.join(model_checkpoint_path, 'Data') 
    # aidata = AIData.load_aidata(data_path, OneAIDataName, SPACE)
    entry = EntryAIData_Builder.load_entry(model_checkpoint_path, SPACE)

    # --------------- loald Model Instance Args ----------------
    with open(os.path.join(model_checkpoint_path, 'ModelInstanceArgs.json'), 'r') as f:
        ModelInstanceArgs = json.load(f)

    # OneModelJobName = OneModelArtifactArgs['OneModelJobName']
    # OneModelJobArgs = OneModelArtifactArgs['OneModelJobArgs']
    # ModelArgs = ModelInstanceArgs['ModelArgs']
    # TrainingArgs = ModelInstanceArgs['TrainingArgs']
    # InferenceArgs = ModelInstanceArgs['InferenceArgs']
    # EvaluationArgs = ModelInstanceArgs['EvaluationArgs']
    
    ModelInstanceArgs['SPACE'] = SPACE

    model_checkpoint_name = os.path.basename(model_checkpoint_path)
    # ModelNames = {
    #     'model_checkpoint_name': model_checkpoint_name,
    #     'MODEL_ENDPOINT': SPACE['MODEL_ENDPOINT'] # MODEL_VERSION,
    # }
    Model_Info = {
        'entry': entry, 
        'ModelInstanceArgs': ModelInstanceArgs,
        'ModelNames': ModelInstanceArgs['ModelNames'],
    }

    return Model_Info    


def load_model_artifact(model_artifact_path, load_model_instance_from_nn, SPACE):
    Model_Info = load_model_configurations(model_artifact_path, SPACE)   
    # aidata = Model_Info['aidata']
    entry = Model_Info['entry']
    ModelNames = Model_Info['ModelNames']
    OneModelInstanceArgs = Model_Info['ModelInstanceArgs']
    model_type = OneModelInstanceArgs['ModelArgs']['model_type']
    ModelInstanceClass = load_model_instance_from_nn(model_type)

    model_artifact = ModelInstanceClass(entry = entry,
                                        # ModelNames = ModelNames,
                                         # SPACE = SPACE, 
                                        **OneModelInstanceArgs)
    model_artifact.init_model()
    model_artifact.load_model(model_artifact_path)
    return model_artifact



class Model_Base:
    def __init__(self, 
                 aidata_base = None, 
                 # Name_to_ModelInstanceClass = None,
                 # when you want to train a model_job_args list.
                 OneAIDataName_to_ModelJobNames = None, 
                 OneModelJobName_to_OneModelJobArgs = None, 
                 # when you want to load model from modelversion
                 ModelEndpoint_Path = None, 
                 load_model_instance_from_nn = None, 
                 SPACE = None, 
                 ):
        

        self.OneAIDataName_to_ModelJobNames = OneAIDataName_to_ModelJobNames
        # self.Name_to_ModelInstanceClass = Name_to_ModelInstanceClass
        self.load_modelbase_from_modelendpoint(ModelEndpoint_Path, load_model_instance_from_nn, SPACE)
        
        # if ModelEndpoint_Path is not None: 
        #     self.OneAIDataName_to_ModelJobNames = OneAIDataName_to_ModelJobNames
        #     # self.Name_to_ModelInstanceClass = Name_to_ModelInstanceClass
        #     self.load_modelbase_from_modelendpoint(ModelEndpoint_Path, load_model_instance_from_nn, SPACE)
        # else:
        #     self.SPACE = SPACE
        #     self.aidata_base = aidata_base
        #     self.OneAIDataName_to_ModelJobNames = OneAIDataName_to_ModelJobNames
        #     self.load_model_instance_from_nn = load_model_instance_from_nn
        #     self.OneModelJobName_to_OneModelJobArgs = OneModelJobName_to_OneModelJobArgs
        #     self.df_modelartifacts = self.get_df_modelartifacts()



    ########################
    # for inference only.
    ########################
    def load_modelbase_from_modelendpoint(self, ModelEndpoint_Path, load_model_instance_from_nn, SPACE):
        
        ModelEndpoint_ModelPath = os.path.join(ModelEndpoint_Path, 'models')
        self.load_model_instance_from_nn = load_model_instance_from_nn
        self.SPACE = SPACE

        # ----------------------------------------------------------------------------
        Model_Artifact_Name_list = os.listdir(ModelEndpoint_ModelPath)
        # ModelInstancePath_list = []
        ModelArtifactName_to_ModelInfo = {}
        for OneModelArtifactName in Model_Artifact_Name_list:
            # OneDataName_Path = os.path.join(ModelEndpoint_ModelPath, OneDataName)
            # # if not os.path.isdir(ModelUnit_Path): continue
            # assert os.path.isdir(OneDataName_Path), f'{OneDataName_Path} is not a directory.'

            # # OneModelInstanceName = os.path.join(ModelUnit, )
            # for OneSubModelArtifactName in os.listdir(OneDataName_Path):

            # -------------- ModelArtifactName --------------
            # OneModelArtifactName = os.path.join(OneDataName, OneSubModelArtifactName)
            OneModelArtifactPath = os.path.join(ModelEndpoint_ModelPath, OneModelArtifactName)
            # -----------------------------------------------
            
            # ----------------- get detailed name information
            # OneDataVariantName, OneModelJobName, FingerPrint = OneSubModelArtifactName.split('__')
            # OneAIDataName = os.path.join(OneDataName, OneDataVariantName)
            # OneAIDataName, model_type, hashid = OneModelArtifactName.split('__')
            OneAIDataName = OneModelArtifactName.split('__')[0]


            # OneJobName, OneJobVariantName = OneModelJobName.split('.')
            # OneModelInstanceName = OneAIDataName + '__' + OneModelJobName
            # -----------------------------------------------
            
            # -------------- load model_artifact --------------
            try:
                logger.info(f'Load ModelArtifact from {OneModelArtifactPath}\n') 
                model_artifact = load_model_artifact(OneModelArtifactPath, load_model_instance_from_nn, SPACE)

            
                # -----------------------------------------------
                
                # -------------- ModelInfo --------------
                ModelInfo = {
                    # 'OneDataName': OneDataName, # Also the data name.
                    # 'OneDataVariantName': OneDataVariantName, 
                    # 'OneJobName': OneJobName,
                    # 'OneJobVariantName': OneJobVariantName,

                    'OneAIDataName': OneAIDataName,
                    # 'ModelNames': ModelNames,
                    # 'OneModelJobName': OneModelJobName,
                    # 'OneModelInstanceName': OneModelInstanceName,
                    
                    # 'FingerPrint': FingerPrint, # model_type + date + hashid
                    'OneModelArtifactName': OneModelArtifactName,
                    'OneModelArtifactPath': OneModelArtifactPath,

                    'model_artifact': model_artifact,
                }
                # -----------------------------------------------
                # model_artifact.aidata
                # model_artifact.OneModelJobArgs   
                

                ModelArtifactName_to_ModelInfo[OneModelArtifactName] = ModelInfo

            except Exception as e:
                trc = traceback.format_exc()
                message = 'Exception during inference: ' + str(e) + '\n' + trc
                # logger.error(f'Fail to Load ModelInstance from {OneModelArtifactPath}\n with message: {message}\n')
                logger.error(f'Fail to Load ModelInstance from {OneModelArtifactPath}\n\n')
                # model_artifact = None
                # raise Exception(message)
        

        logger.info(f'Load ModelArtifacts from {ModelEndpoint_ModelPath}\n')
        # ----------------------------------------------------------------------------
        self.ModelArtifactName_to_ModelInfo = ModelArtifactName_to_ModelInfo


    ########################
    # for training only.
    ########################
    # def update_AIDataBase(self, aidata_base, update_df_modelinstance = False):
    #     self.aidata_base = aidata_base
    #     if update_df_modelinstance:
    #         self.df_modelartifacts = self.get_df_modelartifacts()

    

    ########################
    # for training only.
    ########################
    
    # def get_df_modelartifacts(self):

    #     aidata_base = self.aidata_base
    #     OneModelJobName_to_OneModelJobArgs = self.OneModelJobName_to_OneModelJobArgs
        
    #     ModelInstanceName_to_ModelArtifactArgs = {}
    #     for OneAIDataName, OneAIDataArgs in aidata_base.OneAIDataName_to_OneAIDataArgs.items():
    #         # for OneModelInstanceArgs in OneModelInstanceArgs_list:
    #         for OneModelJobName, OneModelJobArgs in OneModelJobName_to_OneModelJobArgs.items():

    #             OneModelInstanceName = OneAIDataName + '__' + OneModelJobName
    #             # OneModelArtifactName = OneModelInstanceName + '__' + FingerPrint

    #             model_type = OneModelJobArgs['ModelArgs']['model_type']
    #             OneModelArtifactArgs = {
    #                 'OneAIDataName': OneAIDataName,
    #                 'OneAIDataArgs': OneAIDataArgs,
    #                 'model_type': model_type,
    #                 'OneModelJobName': OneModelJobName,
    #                 'OneModelJobArgs': OneModelJobArgs,
    #             }
    #             # ModelInstanceSettings_list.append(ModelInstanceSettings)
    #             ModelInstanceName_to_ModelArtifactArgs[OneModelInstanceName] = OneModelArtifactArgs

    #     df_modelartifacts = pd.DataFrame(ModelInstanceName_to_ModelArtifactArgs).T
    #     self.df_modelartifacts = df_modelartifacts
    #     return df_modelartifacts
    

    # def init_one_modelartifact(self, 
    #                            OneModelArtifactArgs = None, 
    #                            aidata = None, 
    #                            SPACE = None):

    #     if SPACE is None: SPACE = self.SPACE

    #     if aidata is None and 'aidata' not in OneModelArtifactArgs:
    #         aidata_base = self.aidata_base
    #         OneAIDataName = OneModelArtifactArgs['OneAIDataName']
    #         aidata = aidata_base.get_aidata_from_OneAIDataName(OneAIDataName)
    #     else:
    #         aidata = OneModelArtifactArgs['aidata']
        

    #     model_type = OneModelArtifactArgs['model_type']
    #     OneModelJobName = OneModelArtifactArgs['OneModelJobName']
    #     ModelInstance = self.load_model_instance_from_nn(model_type)
        
    #     OneModelJobArgs = OneModelArtifactArgs['OneModelJobArgs']
    #     model_artifact = ModelInstance(
    #         aidata = aidata,
    #         SPACE = SPACE, 
    #         OneModelJobName = OneModelJobName,
    #         **OneModelJobArgs,
    #     )
    #     return model_artifact 
    
    
    # def fit_one_modelartifact(self, model_artifact):
    #     model_artifact.init_model()
    #     model_artifact.fit()
    #     model_artifact.evaluate()
    #     model_artifact.save_model()
    #     return model_artifact
    

    # def fit_all_modelartifacts(self):
    #     # to update this later.
    #     df_modelartifacts = self.df_modelartifacts


    #     ModelArtifactName_to_ModelInfo = {}
    #     # idx_to_modelinstanceinfo = {}
    #     for OneModelInstanceName, OneModelArtifactArgs in df_modelartifacts.iterrows():
    #         model_artifact = self.init_one_modelartifact(OneModelArtifactArgs = OneModelArtifactArgs)
    #         model_artifact = self.fit_one_modelartifact(model_artifact)

    #         model_artifact_name = model_artifact.model_artifact_name
    #         ModelInfo = {
    #             'OneModelInstanceName': OneModelInstanceName,
    #             'model_artifact_name': model_artifact_name,
    #             'model_artifact': model_artifact,
    #         }   
    #         ModelArtifactName_to_ModelInfo[model_artifact_name] = ModelInfo # [idx] = {'model_instance': model_instance, 'modelinstance_info': modelinstance_info}


    #     self.ModelArtifactName_to_ModelInfo = ModelArtifactName_to_ModelInfo
    #     return ModelArtifactName_to_ModelInfo
    

    