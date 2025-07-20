import evaluate
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import GenerationConfig
import logging

logger = logging.getLogger(__name__)

def process_a_single_batch(model, batch, InferenceArgs = None):

    if InferenceArgs is None: InferenceArgs = {}

    # ------------ next-token-generation part ----------------
    NTP_Args = InferenceArgs.get('NTP_Args', None)
    if NTP_Args is not None:
        ###############################
        num_first_tokens_for_ntp = NTP_Args['num_first_tokens_for_ntp']
        items_list = NTP_Args['items_list']
        ###############################
        batch_ntp = {k: v[:, :num_first_tokens_for_ntp] for k, v in batch.items()}
        output = model(**batch_ntp)

        # get predicted_labels
        lm_logits = output.logits

        # get the loss each token
        labels = batch['labels'][:, :num_first_tokens_for_ntp]
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        logits_permuted = shift_logits.permute(0, 2, 1)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        losses = loss_fn(logits_permuted, shift_labels)

        batch_ntp_output = {}

        if 'losses_each_seq' in items_list:
            losses_each_seq = losses.mean(dim=1).detach().cpu().numpy().tolist()
            batch_ntp_output['losses_each_seq'] = losses_each_seq

        if 'losses_each_token' in items_list:
            losses_each_token = losses.detach().cpu().numpy()
            losses_each_token = [losses_each_token[i] for i in range(len(losses_each_token))]
            batch_ntp_output['losses_each_token'] = losses_each_token

        if 'predicted_ntp_labels' in items_list:
            predicted_ntp_labels = torch.argmax(lm_logits, dim=-1)
            predicted_ntp_labels = predicted_ntp_labels.detach().cpu().numpy()# .tolist()
            predicted_ntp_labels = [predicted_ntp_labels[i] for i in range(len(predicted_ntp_labels))]
            batch_ntp_output['predicted_ntp_labels'] = predicted_ntp_labels
    else:
        batch_ntp_output = {}
        

    # ------------ generation part ----------------
    GEN_Args = InferenceArgs.get('GEN_Args', None)
    if GEN_Args is not None:
        ###############################
        items_list = GEN_Args['items_list']
        num_first_tokens_for_gen = GEN_Args['num_first_tokens_for_gen']
        max_new_tokens = GEN_Args['max_new_tokens']
        do_sample = GEN_Args['do_sample']

        # items_list = ['hist', 'real', 'pred', 'logit_scores',]
        # num_first_tokens_for_gen = 289
        # max_new_tokens = 24 
        # do_sample = False 
        ###############################


        HF_GenerationConfig = {}
        HF_GenerationConfig['max_new_tokens'] = max_new_tokens
        HF_GenerationConfig['do_sample'] = do_sample
        HF_GenerationConfig['return_dict_in_generate'] = True
        if 'logit_scores' in items_list:
            HF_GenerationConfig['output_scores'] = True

        batch_gen = {k: v[:, :num_first_tokens_for_gen] for k, v in batch.items()}
        generation_config = GenerationConfig(**HF_GenerationConfig)
        outputs_dict = model.generate(generation_config = generation_config, 
                                    **batch_gen)
        
        batch_gen_output = {}
        # if 'hist' in 
        if 'hist' in items_list:
            hist = batch_gen['input_ids']
            hist = hist.cpu().numpy()
            batch_gen_output['hist'] = hist

        if 'real' in items_list:
            real = batch['labels'][:, num_first_tokens_for_gen: num_first_tokens_for_gen+max_new_tokens]
            real = real.cpu().numpy()
            batch_gen_output['real'] = real

        if 'pred' in items_list:
            sequences = outputs_dict['sequences']
            pred = sequences[:, -max_new_tokens:]
            pred = pred.cpu().numpy()
            batch_gen_output['pred'] = pred

        if 'logit_scores' in items_list:
            logits = outputs_dict['scores']
            logit_scores = np.array([logit.cpu().numpy() 
                                    for logit in logits]
                                    ).transpose(1, 0, 2) 
            batch_gen_output['logit_scores'] = logit_scores

        batch_gen_output = {
            k: [v[i] for i in range(v.shape[0])] for k, v in batch_gen_output.items()
        }

        # logger.info(f'batch_gen_output: {batch_gen_output}')
    else:
        batch_gen_output = {}

    batch_output = {**batch_ntp_output, **batch_gen_output}
    return batch_output


def compute_metrics_for_ntp(eval_preds, experiment_id, AfTknNum = 24):

    metric_acc = evaluate.load("accuracy", experiment_id = experiment_id)
    metric_mse = evaluate.load('mse',      experiment_id = experiment_id)

    preds, labels = eval_preds
    # print(preds.shape, labels.shape)
    # print(preds.shape, labels.shape)
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:]
    preds  = preds[:, :-1] 
    # print(preds.shape, labels.shape)

    all_labels = labels.reshape(-1)
    all_preds = preds.reshape(-1)
    # print(all_labels.shape, all_preds.shape)


    af_labels = labels[:, -AfTknNum:].reshape(-1)
    af_preds  = preds[:, -AfTknNum:].reshape(-1)
    # print(af_labels.shape, af_preds.shape)
    
    d_accu = metric_acc.compute(predictions=all_preds, references=all_labels)
    d_mse = metric_mse.compute(predictions=all_preds, references=all_labels)
    d_accu_af = metric_acc.compute(predictions=af_preds, references=af_labels)
    d_mse_af = metric_mse.compute(predictions=af_preds, references=af_labels)
    
    d = {}
    for k, v in d_accu.items(): d[k] = v
    for k, v in d_accu_af.items(): d[k + '_af'] = v

    for k, v in d_mse.items(): d[k] = v
    for k, v in d_mse_af.items(): d[k + '_af'] = v

    d['rMSE'] = np.sqrt(d['mse'])
    d['rMSEaf'] = np.sqrt(d['mse_af'])
    
    d['ACUU']   = d['accuracy'] # np.sqrt()
    d['ACUUaf'] = d['accuracy_af'] # np.sqrt()
    
    del d['mse'], d['mse_af'], d['accuracy'], d['accuracy_af']
    return d


def transform_for_generation(self, examples):
    input_ids = [i[:-24] for i in examples['input_ids']]
    labels = [i[-24:] for i in examples['labels']]
    # labels = examples['labels'] # [i[-24:] for i in ]
    attention_mask = [[1] * len(i) for i in input_ids]
    d = {
        'input_ids': input_ids, 
        'labels': labels, 
        'attention_mask': attention_mask
        }
    return d

    
def compute_metrics_generation(eval_preds, experiment_id):
    metric_acc = evaluate.load("accuracy", experiment_id = experiment_id)
    metric_mse = evaluate.load('mse', experiment_id = experiment_id)

    preds, labels = eval_preds
    print(preds.shape, labels.shape)
    preds = preds[:, -labels.shape[-1]:]
    print(preds.shape, labels.shape)

    d = {}
    for pred_horizon in [6, 12, 24]:
        d_accu = metric_acc.compute(predictions=preds[:, :pred_horizon].reshape(-1), 
                                    references=labels[:, :pred_horizon].reshape(-1))
        d_mse = metric_mse.compute(predictions=preds[:, :pred_horizon].reshape(-1),  
                                references=labels[:, :pred_horizon].reshape(-1))
        
        for k, v in d_accu.items(): d[k + f'_ph{pred_horizon}'] = v
        for k, v in d_mse.items():  d[k + f'_ph{pred_horizon}'] = v
        d['rmse'+ f'_ph{pred_horizon}'] = np.sqrt(d['mse'+ f'_ph{pred_horizon}'])
    return d

