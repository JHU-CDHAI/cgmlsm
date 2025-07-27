import evaluate
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import torch
from transformers import GenerationConfig
from scipy.special import rel_entr
from scipy.stats import entropy

def convert_cgm2region(x):
    categories = []
    for val in x:
        if val < 54:
            categories.append('VeryLow')
        elif val >= 54 and val < 70:
            categories.append('Low')
        elif val >= 70 and val <= 180:
            categories.append('TIR')
        elif val > 180 and val <= 250:
            categories.append('High')
        elif val > 250:
            categories.append('VeryHigh')


        # elif val > 180:
        #     categories.append('TAR')
        # elif val < 70:
        #     categories.append('TBR')


    return categories

from nn.eval.seqeval_helper import SeqEvalHelper

class SeqEvalForOneDataPoint:
    def __init__(self, 
                 x_obs_seq,
                 y_real_seq,
                 y_pred_seq,
                 losses_each_seq = None, 
                 losses_each_token = None, 
                 metric_list = None, 
                 ):
        self.x_obs_seq  = x_obs_seq
        self.y_real_seq = y_real_seq
        self.y_pred_seq = y_pred_seq
        self.metric_list = metric_list
        self.losses_each_seq = losses_each_seq
        self.losses_each_token = losses_each_token
        # print(x_obs_seq.shape, y_real_seq.shape, y_pred_seq.shape)
        self.seqeval_helper = SeqEvalHelper(x_obs_seq, y_real_seq, y_pred_seq, metric_list)


    def cal_losses_each_seq(self):
        return self.losses_each_seq
    

    def get_metric_scores(self):
        metric_func = {
            # 'MSE': cal_MSE, 
            'rMSE': self.cal_rMSE,
            'MAE':  self.cal_MAE,
            'MAEin5': self.cal_MAEin5, 
            'MAEin10': self.cal_MAEin10, 
            'RegionAccu': self.cal_region_accuracy,
            'RegionAccuByRegion': self.cal_region_accuracy_by_region , 
            'InputFeature': self.cal_x_obs_seq_feature,
            # 'RealVar': cal_realVar
            # 'GenToRealDiff': self.cal_pred_to_real_diff
            # 'loss': self.cal_losses_each_seq,
            'RegionDistError': self.compute_distribution_error,
            'TIR_error': self.compute_TIR_error,
            'polynomial': self.seqeval_helper.get_metric_scores_func('polynomial'),
            'entropy':self.seqeval_helper.get_metric_scores_func('entropy'),
            'gmm':self.seqeval_helper.get_metric_scores_func('gmm'),
            'stl':self.seqeval_helper.get_metric_scores_func('stl'),
            'autocorr':self.seqeval_helper.get_metric_scores_func('autocorr'),
            'frechet':self.seqeval_helper.get_metric_scores_func('frechet'),
            'kl':self.seqeval_helper.get_metric_scores_func('kl'),
        }
        if self.metric_list is None:
            self.metric_list = [i for i in metric_func.keys()]


        x_obs_seq = self.x_obs_seq
        y_real_seq = self.y_real_seq
        y_pred_seq = self.y_pred_seq

        metric_scores = {}
        for metric, func in metric_func.items():
            
            if metric not in self.metric_list: continue 

            # print('==============')
            # print(metric)
            # metric_score = func(y_real_seq, y_pred_seq)
            try:
                metric_score = func(x_obs_seq, y_real_seq, y_pred_seq)
            except Exception as e:
                print(f"Error calculating {metric}: {e}")
                print(y_real_seq, y_pred_seq)
                metric_score = None

            if type(metric_score) == dict:
                for k, v in metric_score.items():
                    metric_name = f'{metric}_{k}'
                    metric_scores[metric_name] = v
            else:
                metric_name = f'{metric}'
                metric_scores[metric_name] = metric_score

        # metric_scores = {k: round(v, 5) for k, v in metric_scores.items()}
        return metric_scores



    def cal_region_accuracy(self, x_obs_seq, y_real_seq, y_pred_seq):
        # one datapoint
        y_real_region_seq = convert_cgm2region(y_real_seq)
        y_pred_region_seq = convert_cgm2region(y_pred_seq)
        if len(y_pred_region_seq) != len(y_real_region_seq):
            raise ValueError("Lists must be of the same length")
        matches = sum(l == g for l, g in zip(y_real_region_seq, y_pred_region_seq))
        region_accuracy = matches / len(y_pred_region_seq)
        return region_accuracy 
    
    def cal_region_accuracy_by_region(self, x_obs_seq, y_real_seq, y_pred_seq):
        # one datapoint
        y_real_region_seq = convert_cgm2region(y_real_seq)
        y_pred_region_seq = convert_cgm2region(y_pred_seq)

        if len(y_real_region_seq) != len(y_pred_region_seq):
            raise ValueError("Lists must be of the same length")

        df = pd.DataFrame({'pred': y_pred_region_seq, 'real': y_real_region_seq})
        df['match'] = df['pred'] == df['real']
        d = df.groupby('real')['match'].mean().to_dict()
        return d
    
    ############

    # def cal_region_accuracy(self, x_obs_seq, y_real_seq, y_pred_seq):
    #     # one datapoint
    #     y_real_region_seq = convert_cgm2region(y_real_seq)
    #     y_pred_region_seq = convert_cgm2region(y_pred_seq)
    #     if len(y_pred_region_seq) != len(y_real_region_seq):
    #         raise ValueError("Lists must be of the same length")
            
    #     # Get value counts for both sequences
    #     real_counts = pd.Series(y_real_region_seq).value_counts().sort_index()
    #     pred_counts = pd.Series(y_pred_region_seq).value_counts().sort_index()
        
    #     # Calculate rates (proportions)
    #     real_rates = real_counts / len(y_real_region_seq)
    #     pred_rates = pred_counts / len(y_pred_region_seq)
    #     return distribution_similarity

    def compute_distribution_error(self, x_obs_seq, y_real_seq, y_pred_seq):
        # Convert to numpy arrays for faster computation
        true_vals = np.array(y_real_seq)
        pred_vals = np.array(y_pred_seq)
        
        # Define bins and labels
        bins = np.array([0, 54, 70, 180, 250, np.inf])
        labels = np.array(['hypo_l2', 'hypo_l1', 'in_range', 'hyper_l1', 'hyper_l2'])
        
        # Calculate histograms directly using numpy
        true_counts = np.histogram(true_vals, bins=bins)[0]
        pred_counts = np.histogram(pred_vals, bins=bins)[0]
        
        # Normalize counts
        true_dist = true_counts / len(true_vals)
        pred_dist = pred_counts / len(pred_vals)
        
        # Calculate metrics
        l1 = np.abs(true_dist - pred_dist).sum()
        l2 = np.sqrt(((true_dist - pred_dist) ** 2).sum())
        kl = np.sum(rel_entr(true_dist + 1e-8, pred_dist + 1e-8))
        
        return {
            'l1': l1,
            'l2': l2,
            'kl': kl
        }
    #############

    def compute_TIR_error(self, x_obs_seq, y_real_seq, y_pred_seq):
        # Convert to numpy arrays for faster computation
        true_vals = np.array(y_real_seq)
        pred_vals = np.array(y_pred_seq)
        
        # Define bin edges
        bins = np.array([0, 54, 70, 180, 250, np.inf])
        
        # Calculate bin counts directly using numpy
        true_counts = np.histogram(true_vals, bins=bins)[0]
        pred_counts = np.histogram(pred_vals, bins=bins)[0]
        
        # Calculate TIR (Time in Range) for both sequences
        true_TIR = true_counts[2] / len(true_vals)  # index 2 corresponds to in_range bin
        pred_TIR = pred_counts[2] / len(pred_vals)
        
        return np.abs(true_TIR - pred_TIR)
    

    def cal_MSE(self, x_obs_seq, y_real_seq, y_pred_seq):
        MSE = np.mean((np.array(y_real_seq) - np.array(y_pred_seq))**2)
        return MSE


    def cal_rMSE(self, x_obs_seq, y_real_seq, y_pred_seq):
        # Squared_Error = np.mean((np.array(gen) - np.array(real))**2)
        MSE = np.mean((np.array(y_real_seq) - np.array(y_pred_seq))**2)
        rMSE = np.round(np.sqrt(MSE), 2)
        # print(Squared_Error)
        # MSE = Squared_Error / 24
        # rMSE = np.round(np.sqrt(Squared_Error), 2)

        return rMSE

    def cal_MAE(self, x_obs_seq, y_real_seq, y_pred_seq):
        # Abs_Error = np.mean(np.abs(np.array(gen) - np.array(real)))
        # print(Squared_Error)
        # MSE = Squared_Error / 24
        # return np.round(Abs_Error, 2)
        # MAE = np.mean(np.abs(np.array(gen) - np.array(real)))
        MAE = np.mean(np.abs(np.array(y_real_seq) - np.array(y_pred_seq)))
        return np.round(MAE, 2)
    

    def cal_MAEin5(self, x_obs_seq, y_real_seq, y_pred_seq):
        tolerance = 5
        # Abs_Error = np.abs(np.array(gen) - np.array(real))
        # Within5 = (Abs_Error <= tolerance).mean()
        # return np.round(Within5, 2)
        MAE = np.abs(np.array(y_real_seq) - np.array(y_pred_seq))
        MAEin = (MAE <= tolerance).mean()
        return np.round(MAEin, 2)

    def cal_MAEin10(self, x_obs_seq, y_real_seq, y_pred_seq):
        tolerance = 10
        MAE = np.abs(np.array(y_real_seq) - np.array(y_pred_seq))
        MAEin = (MAE <= tolerance).mean()
        return np.round(MAEin, 2)
    
    def cal_realVar(self, x_obs_seq, y_real_seq, y_pred_seq):
        return np.var(y_real_seq)

    def cal_pred_to_real_diff(self, x_obs_seq, y_real_seq, y_pred_seq):
        return np.array(y_pred_seq) - np.array(y_real_seq)



    def cal_x_obs_seq_feature(self, x_obs_seq, y_real_seq, y_pred_seq):
        # return x_obs_seq
        cgm_sequence = np.array(x_obs_seq)
        diffs = np.diff(cgm_sequence)
        normed = (cgm_sequence - np.min(cgm_sequence)) / (np.max(cgm_sequence) - np.min(cgm_sequence) + 1e-5)

        features = {
            "mean_glucose": np.mean(cgm_sequence),
            "std_glucose": np.std(cgm_sequence),
            "range_glucose": np.max(cgm_sequence) - np.min(cgm_sequence),
            "coef_var": np.std(cgm_sequence) / (np.mean(cgm_sequence) + 1e-5),
            "mean_roc": np.mean(np.abs(diffs)),
            "max_roc": np.max(np.abs(diffs)),
            "tir": np.mean((cgm_sequence >= 70) & (cgm_sequence <= 180)),
            "tar": np.mean(cgm_sequence > 180),
            "tbr": np.mean(cgm_sequence < 70)
        }
        return features

    

    def plot_cgm_sensor(self, x_obs_seq = None, y_real_seq = None, y_pred_seq = None):
        if x_obs_seq is None:
            x_obs_seq = self.x_obs_seq
        if y_real_seq is None:
            y_real_seq = self.y_real_seq
        if y_pred_seq is None:
            y_pred_seq = self.y_pred_seq

        # gen = example[f'pred_{max_new_tokens}'].tolist()
        # real = example[f'real_{max_new_tokens}'].tolist()
        obs  = x_obs_seq
        gen  = y_pred_seq #  example[gen_id_col]# .tolist()
        real = y_real_seq # example[real_id_col]# .tolist()
        
        
        
        plt.figure(figsize=(10, 6))
        # Assuming hypothetical TIR thresholds for demonstration
        O = 0
        A = 54
        B = 70
        C = 180
        D = 250
        E = 300

        plt.axhline(y=O, color='g', linestyle='--',  linewidth = 1.5, alpha = 0.5)
        plt.axhline(y=A, color='g', linestyle='--', label='54',  linewidth = 1.5, alpha = 0.5)
        plt.axhline(y=B, color='g', linestyle='--', label='70',  linewidth = 1.5, alpha = 0.5)
        plt.axhline(y=C, color='g', linestyle='--', label='180', linewidth = 1.5, alpha = 0.5)
        plt.axhline(y=D, color='g', linestyle='--', label='250', linewidth = 1.5, alpha = 0.5)
        plt.axhline(y=E, color='g', linestyle='--', label='300', linewidth = 1.5, alpha = 0.5)

        print('gen:', gen)
        print('real:', real)
        plt.plot(gen, label='Generated', marker='o')
        plt.plot(real, label='Actual', linestyle='--', marker='x')
        plt.title('Comparison of Generated and Actual Series')
        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        plt.legend()
        # plt.grid(True)
        plt.show()
        
        
        # gen = example[input_id_cols][:289].tolist() + example[f'gen_{max_new_tokens}'].tolist()
        # real = example[input_id_cols][:289].tolist() + example[f'label_{max_new_tokens}'].tolist()
        
        gen  = list(obs) + list(y_pred_seq)
        real = list(obs) + list(y_real_seq)

        plt.figure(figsize=(10, 6))

        plt.axhline(y=O, color='g', linestyle='--',  linewidth = 1.5, alpha = 0.5)
        plt.axhline(y=A, color='g', linestyle='--', label='54',  linewidth = 1.5, alpha = 0.5)
        plt.axhline(y=B, color='g', linestyle='--', label='70',  linewidth = 1.5, alpha = 0.5)
        plt.axhline(y=C, color='g', linestyle='--', label='180', linewidth = 1.5, alpha = 0.5)
        plt.axhline(y=D, color='g', linestyle='--', label='250', linewidth = 1.5, alpha = 0.5)
        plt.axhline(y=E, color='g', linestyle='--', label='300', linewidth = 1.5, alpha = 0.5)


        plt.plot(gen, label='Generated', marker='o')
        plt.plot(real, label='Actual', linestyle='--', marker='x')
        plt.title('Comparison of Generated and Actual Series')
        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        plt.ylim(0, 401)
        plt.legend()
        # plt.grid(True)
        plt.show()

    
class SeqEvalForOneDataPointWithHorizons:
    def __init__(self,
                 x_obs_seq_total,
                 y_real_seq_total,
                 y_pred_seq_total,
                 metric_list,
                 horizon_to_se,
                 # losses_each_token_total = None,
                ):
        self.x_obs_seq_total = x_obs_seq_total
        self.y_real_seq_total = y_real_seq_total
        self.y_pred_seq_total = y_pred_seq_total
        self.metric_list = metric_list
        self.horizon_to_se = horizon_to_se
        # self.losses_each_token_total = losses_each_token_total
    
    def get_complete_metrics_with_horizon(self):
        horizon_to_se = self.horizon_to_se
        x_obs_seq_total = self.x_obs_seq_total
        y_real_seq_total = self.y_real_seq_total
        y_pred_seq_total = self.y_pred_seq_total
        metric_list = self.metric_list
        
        metric_total = {}
        for horizon, se in horizon_to_se.items():
            start, end = se
            y_pred_seq = y_pred_seq_total[start:end]
            y_real_seq = y_real_seq_total[start:end]
            metric = SeqEvalForOneDataPoint(
                x_obs_seq_total, 
                y_real_seq, 
                y_pred_seq, 
                metric_list = metric_list,
                # losses_each_seq = self.losses_each_seq,
            ).get_metric_scores()
            # print(metric)
            for k, v in metric.items():
                metric_total[f'{horizon}_{k}'] = v
        self.metric_total = metric_total
        return metric_total


class SeqEvalForOneEvalSet:
    def __init__(self, 
                 setname, 
                 df_case_eval, 
                 x_hist_seq_name,
                 y_real_seq_name, 
                 y_pred_seq_name,
                 metric_list,
                 horizon_to_se, 
                 ):
        print(metric_list)
        # self.EvaluationArgs= EvaluationArgs
        self.setname = setname
        self.df_case_eval = df_case_eval
        self.metric_list = metric_list

        self.y_real_seq_name = y_real_seq_name
        self.y_pred_seq_name = y_pred_seq_name
        self.x_hist_seq_name = x_hist_seq_name

        self.horizon_to_se = horizon_to_se
        # self.y_real_seq_array = df_case_eval[y_real_seq_name]
        # self.y_pred_seq_array = df_case_eval[y_pred_seq_name]
        # self.x_hist_seq_array = df_case_eval[x_hist_seq_name]

    def calculate_stats(self, df, metric_list, confidence=0.95, method='bootstrap', n_bootstrap=1000):
        """
        Calculate mean and confidence interval for each column in the dataframe using either t-test or bootstrap method
        
        Parameters:
        df (pandas.DataFrame): Input dataframe
        metric_list (list): List of metric columns to calculate statistics for
        confidence (float): Confidence level (default 0.95 for 95% confidence interval)
        method (str): Method to calculate CI - either 't-test' or 'bootstrap'
        n_bootstrap (int): Number of bootstrap samples (only used if method='bootstrap')
        
        Returns:
        dict: Dictionary containing mean and confidence intervals for each column
        """
        results = []
        
        for column in metric_list:
            mean = df[column].mean()
            
            if method == 't-test':
                # Option 1: Using t-test
                n = len(df[column])
                std_err = stats.sem(df[column])
                ci = stats.t.interval(confidence, n-1, loc=mean, scale=std_err)
                
            elif method == 'bootstrap':
                # Option 2: Using bootstrap
                bootstrap_means = []
                for _ in range(n_bootstrap):
                    # Sample with replacement
                    bootstrap_sample = np.random.choice(df[column], size=len(df[column]), replace=True)
                    bootstrap_means.append(np.mean(bootstrap_sample))
                
                # Calculate confidence interval from bootstrap distribution
                ci_lower = np.percentile(bootstrap_means, ((1 - confidence) / 2) * 100)
                ci_upper = np.percentile(bootstrap_means, (1 - (1 - confidence) / 2) * 100)
                ci = (ci_lower, ci_upper)
            else:
                ci = (mean, mean)
                # raise ValueError(f"Invalid method: {method}")
                print(f"Invalid method: {method}")
            
            std = np.std(df[column])

            results.append({
                'Column': column,
                'Mean': mean,
                'Std': std,
                'CI_Lower': ci[0],
                'CI_Upper': ci[1],
                'CI_Range': f'({ci[0]:.2f}, {ci[1]:.2f})'
            })
        
        df = pd.DataFrame(results)
        df = df.set_index('Column').T

        d1 = df.loc['Mean'].to_dict()
        d2 = {k+':all': v for k, v in df.to_dict(orient='dict').items()}

        d = {**d1, **d2}
        return d

    
    def get_df_case_metric(self, 
                             df_case_eval = None,
                             x_hist_seq_name = None,
                             y_real_seq_name = None, 
                             y_pred_seq_name = None,
                             metric_list = None,
                             horizon_to_se = None):
        
        if df_case_eval is None:
            df_case_eval = self.df_case_eval    
        if x_hist_seq_name is None:
            x_hist_seq_name = self.x_hist_seq_name
        if y_real_seq_name is None:
            y_real_seq_name = self.y_real_seq_name
        if y_pred_seq_name is None:
            y_pred_seq_name = self.y_pred_seq_name
        if metric_list is None:
            metric_list = self.metric_list
        if horizon_to_se is None:
            horizon_to_se = self.horizon_to_se

        metric_array = df_case_eval.apply(
            lambda x: SeqEvalForOneDataPointWithHorizons(x[x_hist_seq_name], 
                                                        x[y_real_seq_name], 
                                                        x[y_pred_seq_name], 
                                                        metric_list  = metric_list,
                                                        horizon_to_se = horizon_to_se).get_complete_metrics_with_horizon(),
            axis = 1
        )

        df_metric = pd.DataFrame(metric_array.to_list())
        metric_list = df_metric.columns

        df_case_metric = pd.concat([df_case_eval, df_metric], axis=1).reset_index(drop = True)
        self.df_case_metric = df_case_metric
        self.metric_list = metric_list  

        return df_case_metric, metric_list
    

    def get_evaluation_report(self, metric_list = None, method = 't-test', confidence=0.95):
        # print(metric_list)
        df_case_metric, metric_list = self.get_df_case_metric(metric_list = metric_list)
    
        metric_scores = self.calculate_stats(df_case_metric, metric_list, method = method, confidence=confidence)
        return metric_scores

    
class SeqPredEval:
    
    def __init__(self, 
                 df_case_eval, 
                 subgroup_config_list, 
                 x_hist_seq_name,
                 y_real_seq_name, 
                 y_pred_seq_name,
                 metric_list,
                 horizon_to_se,
                 method = 't-test',
                 confidence=0.95):
        
        self.df_case_eval = df_case_eval
        self.subgroup_config_list = subgroup_config_list
        self.x_hist_seq_name = x_hist_seq_name
        self.y_real_seq_name = y_real_seq_name
        self.y_pred_seq_name = y_pred_seq_name
        self.metric_list = metric_list
        self.horizon_to_se = horizon_to_se

        report_list = []
        for subgroup_config in subgroup_config_list:
            for setname, df_case_eval_by_group in df_case_eval.groupby(subgroup_config):
                # print(group_name)
                # SetName_to_DfCaseEval[SetName] = df_case_eval_by_group

                eval_instance = SeqEvalForOneEvalSet(
                    setname = setname,
                    df_case_eval = df_case_eval_by_group, 
                    x_hist_seq_name = x_hist_seq_name,
                    y_real_seq_name = y_real_seq_name, 
                    y_pred_seq_name = y_pred_seq_name,
                    metric_list = metric_list,
                    horizon_to_se = horizon_to_se, 
                )
                # get_evaluation_report(self, method = 't-test', confidence=0.95)
                # print(metric_list)
                eval_results = eval_instance.get_evaluation_report(metric_list = metric_list, method = method, confidence=confidence)
                eval_results['setname'] = setname
                report_list.append(eval_results)

        df_report_full = pd.DataFrame(report_list)

        column_prefix_columns = ['setname']
        columns = column_prefix_columns + [i for i in df_report_full.columns if i not in column_prefix_columns]
        df_report_full = df_report_full[columns].reset_index(drop = True)

        columns_dfreport = [i for i in df_report_full.columns if ':all' not in i]
        df_report_neat = df_report_full[columns_dfreport].reset_index(drop = True)  
        self.df_report_full = df_report_full
        self.df_report_neat = df_report_neat
