import os
import pandas as pd
import numpy as np
from datetime import datetime 
from datasets.fingerprint import Hasher 

from tqdm import tqdm
import torch
import json 
import logging
from recfldtkn.aidata_base.aidata import AIData

from .configuration_cgmlsm import CgmGptConfig
from .modeling_cgmlsm import CgmGptLMHeadModel, CgmGptDistHeadModel
from .inference_cgmlsm import (
    compute_metrics_for_ntp,
    process_a_single_batch,
)
from ..eval.seqeval import SeqPredEval

from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint
from ..base import DeepLearningInstance

logger = logging.getLogger(__name__)


def prepare_last_checkpoint(training_args):
    # ------------------------------- part 3: last checkpoint -------------------------------
    # Detecting last checkpoint.
    last_checkpoint = None

    dont_overwrite_output_dir = bool(not training_args.overwrite_output_dir)

    if os.path.isdir(training_args.output_dir) and training_args.do_train and dont_overwrite_output_dir:

        last_checkpoint = get_last_checkpoint(training_args.output_dir)

        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
               f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
               f"Checkpoint detected, resuming training at {last_checkpoint}."
                "To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    return last_checkpoint


class TimestampCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Add the current timestamp to the logs
        logs["step"] = state.global_step
        logs["timestamp"] = str(datetime.now())


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    # print(logits.shape, type(logits), '<----- logits')
    return logits.argmax(dim=-1)


class CgmGptInstance(DeepLearningInstance):

    def __init__(self, 
                 aidata = None,
                 ModelArgs = None, 
                 TrainingArgs = None, 
                 InferenceArgs = None, 
                 EvaluationArgs = None,
                 OneModelJobName = 'base.v1',
                 SPACE = None, 
                 ):
        
        if aidata is not None:
            # ----- within the method of init_model.
            CF_to_CFvocab = aidata.CF_to_CFvocab
            CF = list(CF_to_CFvocab.keys())[0]
            CFvocab = CF_to_CFvocab[CF]
            tkn2tid = CFvocab['input_ids']['tkn2tid']
            config_kwargs = {
                ###########
                'vocab_size': len(tkn2tid),
                'bos_token_id': tkn2tid['[BOS]'],
                'eos_token_id': tkn2tid['[EOS]'],
                'pad_token_id':  0,
                ###########
            }
            ModelArgs.update(config_kwargs)

        self.aidata = aidata
        self.ModelArgs = ModelArgs
        self.TrainingArgs = TrainingArgs
        self.InferenceArgs = InferenceArgs
        self.EvaluationArgs = EvaluationArgs
        self.SPACE = SPACE

        
        self.OneModelJobName = OneModelJobName
        self.OneModelArtifactArgs = self.get_OneModelArtifactArgs()
        self.model_artifact_name = self.get_model_artifact_name()
        self.model_artifact_path = self.get_model_artifact_path()

    def init_model(self):

        config = CgmGptConfig(**self.ModelArgs)
        set_seed(config.seed)
        self.config = config

        if self.ModelArgs['model_type'] == 'cgmlsm_lm':
            model = CgmGptLMHeadModel(config)
        elif self.ModelArgs['model_type'] == 'cgmlsm_dist':
            model = CgmGptDistHeadModel(config)
        else:
            raise NotImplementedError
        
        self.model = model
        

    def fit(self, config = None, model = None, aidata = None, TrainingArgs = None):

        # ###############################TrainEvals = aidata.TrainEvals
        if aidata is None: aidata = self.aidata
        if config is None: config = self.config 
        if model is None:  model = self.model
        if TrainingArgs is None: TrainingArgs = self.TrainingArgs
        
        # ######## TrainEvals
        TrainEvals = aidata.TrainEvals
        TrainSetName = TrainEvals['TrainSetName']
        EvalSetNames = TrainEvals['EvalSetNames']
        max_train_samples = TrainingArgs.get('max_train_samples', None)
        max_eval_samples  = TrainingArgs.get('max_eval_samples', None)

        # ######## HuggingFaceTrainingArgs
        HuggingFaceTrainingArgs = TrainingArgs.get('HuggingFaceTrainingArguments', {})
        HuggingFaceTrainingArgs['output_dir'] = self.model_artifact_path


        # ######## AfTknNum
        AfTknNum = TrainingArgs.get('AfTknNum', 24)

        # ------------ train datasets ------------
        TrainData = aidata.Name_to_Data[TrainSetName]
        ds_tfm_train = TrainData['ds_tfm']
        if max_train_samples is not None:
            max_train_samples = min(len(ds_tfm_train), max_train_samples)
            ds_tfm_train = ds_tfm_train.shuffle(seed=42).select(range(max_train_samples))
        logger.info(f'---- train_dataset ----')
        logger.info(ds_tfm_train)

        # ------------ eval datasets ------------
        eval_dataset_dict = {}
        for evalname in EvalSetNames:
            if evalname not in aidata.Name_to_Data: continue
            eval_dataset = aidata.Name_to_Data[evalname]['ds_tfm']
            if max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset), max_eval_samples)
                eval_dataset = eval_dataset.shuffle(seed=42).select(range(max_eval_samples))
            eval_dataset_dict[evalname] = eval_dataset
        logger.info(f'---- eval_datasets ----')
        logger.info(eval_dataset_dict)


        # --------------------- part 1: training args ---------------------
        training_args = TrainingArguments(**HuggingFaceTrainingArgs)
        self.training_args = training_args

        # --------------------- part 2: prepare trainer ---------------------
        set_seed(self.training_args.seed)
        timestamp = datetime.now().strftime("%Y%m%d-%H")
        experiment_id = timestamp + "-" + Hasher().hash([aidata.OneAIDataArgs, config])

        training_args = self.training_args
        trainer = Trainer(
            ########## you have your model 
            model = model,
            ########## you have your training_args
            args = training_args,
            ########## get train_dataset
            train_dataset = ds_tfm_train, # if training_args.do_train else None,
            ########## get eval_dataset
            eval_dataset = eval_dataset_dict, # <--- for in-training evaluation
            ########## huge question here: is it ok to ignore the tokenizer?
            # tokenizer = tokenizer, # Apr 2024: don't add tokenizer, hard to save.
            ########## huge question here: data_collator
            data_collator = default_data_collator,
            compute_metrics = lambda x: compute_metrics_for_ntp(x, experiment_id, AfTknNum),
            preprocess_logits_for_metrics = preprocess_logits_for_metrics,
            callbacks = [TimestampCallback],
        )

        self.trainer = trainer
        logger.info(trainer)

        # --------------------- part 3: train ---------------------
        checkpoint = prepare_last_checkpoint(training_args)
        train_result = trainer.train(resume_from_checkpoint = checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        max_train_samples = max_train_samples if max_train_samples is not None else len(ds_tfm_train)
        metrics["train_samples"] = min(max_train_samples, len(ds_tfm_train))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        self.train_result = train_result
        self.model = trainer.model


    def inference(self, Data, InferenceArgs = None):
        
        ###################################
        if InferenceArgs is None: InferenceArgs = self.InferenceArgs
        # aidata = self.aidata
        model = self.model
        # SPACE = self.SPACE


        ###################################
        max_inference_num = InferenceArgs.get('max_inference_num', None)
        save_df = InferenceArgs.get('save_df', False)
        load_df = InferenceArgs.get('load_df', False)
        chunk_size = InferenceArgs.get('chunk_size', 12800)
        batch_size = InferenceArgs.get('batch_size', 64)
        # case_id_columns = aidata.case_id_columns


        ############ prepare the Data to process ############
        ds_tfm = Data['ds_tfm']
        df_case = Data['df_case']
        if max_inference_num is not None: 
            ds_tfm = ds_tfm.select(range(max_inference_num))
            df_case = df_case.iloc[:max_inference_num]
            

        ########### prepare the chunks to process ############
        chunk_numbers = len(df_case) // chunk_size

        li_df_eval_chunk_path = []
        li_df_eval_chunk_data = []

        for chunk_id in range(chunk_numbers + 1):
            start = chunk_id * chunk_size
            end = min((chunk_id+1) * chunk_size, len(df_case))
            
            
            df_case_chunk = df_case.iloc[start:end].reset_index(drop = True)
            ds_tfm_chunk = ds_tfm.select(range(start, end))
            
            folder = os.path.join(self.model_artifact_path, 'Inference')
            file = os.path.join(folder, f'chunk_{chunk_id:06}_s{start}_e{end}.p')
            
            if save_df == True and not os.path.exists(folder):
                os.makedirs(folder)
                logger.info(f'Creating folder {folder}')

            if load_df == True and os.path.exists(file):
                logger.info(f'Loading chunk {chunk_id} from {file}')
                li_df_eval_chunk_path.append(file)
                continue
            
            df_eval_chunk = pd.DataFrame()
            if len(ds_tfm_chunk) <= 10:
                iterrator_object = range(0, len(ds_tfm_chunk), batch_size)
            else:
                iterrator_object = tqdm(range(0, len(ds_tfm_chunk), batch_size))
                
            for batch_s in iterrator_object:
                batch_e = min(batch_s + batch_size, len(ds_tfm_chunk))
                batch = ds_tfm_chunk[batch_s: batch_e]
                for k, v in batch.items():
                    batch[k] = v.to(model.device)
                with torch.no_grad():
                    model.eval()
                    # logger.info(f'Processing chunk {chunk_id} batch {batch_s} to {batch_e}: {batch}')
                    output = process_a_single_batch(model, batch, InferenceArgs)
                    # logger.info(f'Output: {output}')

                df_batch = pd.DataFrame(output)
                # logger.info(df_batch)
                df_eval_chunk = pd.concat([df_eval_chunk, df_batch], axis = 0)
                # logger.info(df_eval_chunk)
            
            df_eval_chunk = df_eval_chunk.reset_index(drop=True)

            # df_chunk_caseids = dataset_chunk.select_columns(case_id_columns).to_pandas()
            assert len(df_eval_chunk) == len(df_case_chunk), f'{len(df_eval_chunk)} != {len(df_case_chunk)}'
            df_chunk = pd.concat([df_case_chunk, df_eval_chunk], axis = 1)
            df_chunk = df_chunk.reset_index(drop=True)

            if save_df == True:
                logger.info(f'Saving chunk {chunk_id} to {file}')
                df_chunk.to_pickle(file)
                li_df_eval_chunk_path.append(file)
            else:
                li_df_eval_chunk_data.append(df_chunk)

        if save_df == True:
            df_case_eval = pd.concat([pd.read_pickle(file) for file in li_df_eval_chunk_path], axis = 0)
        else:
            df_case_eval = pd.concat(li_df_eval_chunk_data, axis = 0)
        
        results = {
            'df_case_eval': df_case_eval,
        }
        return results

    
    def evaluate(self, 
                 aidata = None, 
                 EvalSetNames = None, 
                 InferenceArgs = None, 
                 EvaluationArgs = None):
        

        ####################################
        if aidata is None: aidata = self.aidata
        if InferenceArgs is None: InferenceArgs = self.InferenceArgs
        if EvaluationArgs is None: EvaluationArgs = self.EvaluationArgs
        if EvalSetNames is None: EvalSetNames = aidata.TrainEvals['EvalSetNames']

        model_instance = self

        df_case_list = []
        for SetName in EvalSetNames:
            logger.info(f'Evaluate on {SetName}...')
            if SetName not in aidata.Name_to_Data: continue 
            Data    = aidata.Name_to_Data[SetName] # Data['df_case'] (meta), Data['ds_tfm'] (CF). 
            df_case = Data['df_case'].copy()
            df_case['EvalName'] = SetName   
            Data['df_case'] = df_case   
            inference_results = model_instance.inference(Data, InferenceArgs)
            
            df_case_eval_oneset = inference_results['df_case_eval']
            df_case_list.append(df_case_eval_oneset)

        df_case_eval = pd.concat(df_case_list)
        self.df_case_eval = df_case_eval

        ############ evalaution. 
        subgroup_config_list = EvaluationArgs['subgroup_config_list']
        x_hist_seq_name      = EvaluationArgs['x_hist_seq_name']
        y_real_seq_name      = EvaluationArgs['y_real_seq_name']
        y_pred_seq_name      = EvaluationArgs['y_pred_seq_name']
        metric_list          = EvaluationArgs['metric_list']
        horizon_to_se        = EvaluationArgs['horizon_to_se']


        eval_instance = SeqPredEval(
            df_case_eval = df_case_eval, 
            subgroup_config_list = subgroup_config_list,
            x_hist_seq_name = x_hist_seq_name,
            y_real_seq_name = y_real_seq_name,
            y_pred_seq_name = y_pred_seq_name,
            metric_list = metric_list,
            horizon_to_se = horizon_to_se
        )

        self.eval_instance  = eval_instance  
        self.df_report_full = eval_instance.df_report_full
        self.df_report_neat = eval_instance.df_report_neat
    
    