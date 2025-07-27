import numpy as np 
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Input: x, predicted a, predicted reward
# 


class BanditPredEvalForOneEvalSet:

    def __init__(self, 
                 setname, 
                 df_case_eval, 
                 action_name, 
                 reward_predicted_name,
                 action_distribution_name,
                 ):
        
        self.setname = setname
        self.df_case_eval = df_case_eval
        self.action_name = action_name
        self.reward_predicted_name = reward_predicted_name
        self.action_distribution_name = action_distribution_name


        self.action = df_case_eval[self.action_name]
        self.pred_reward = df_case_eval[self.reward_predicted_name]
        # self.action_distribution = df_case_eval[self.action_distribution_name]

        # self.GroupNum = GroupNum

    def get_evaluation_report(self):

        report = {'name': self.setname}
        d = self.get_evaluations()
        for k, v in d.items(): report[k] = v

        return report
    
    def get_evaluations(self):
        
        
        SampleNum = len(self.action)
        if SampleNum == 0: 
            d = {
                'SampleNum': 0,
                'RealPosNum': 0, 
                'RealPosRate': None,
                'PredScoreMean': None,
                'PredScoreStd': None,
                'PredScoreMin': None, 
                'PredScoreMax': None,
                'auc': None,
            }
            return d
        
        d = {}

        d['SampleNum'] = len(self.action)
        d['PredEngagement'] = np.mean(self.pred_reward)

        unique_action = np.unique(self.action)
        for i in range(len(np.sort(unique_action))):
            d['Action_' + str(np.sort(unique_action)[i])] = round(np.sum(self.action == np.sort(unique_action)[i]) / len(self.action),4)
        return d
    
        
        
class BanditPredEval:

    def __init__(self, 
                 df_case_eval, 
                 subgroup_config_list, 
                 action_name, 
                 reward_predicted_name,
                 action_distribution_name,
                 ):
        
        # self.SetName_to_DfCaseEval = SetName_to_DfCaseEval
        # for setname, df_case_eval in SetName_to_DfCaseEval.items():
        #     df_case_eval['SetName'] = setname

        report_list = []
        for subgroup_config in subgroup_config_list:
            for setname, df_case_eval_by_group in df_case_eval.groupby(subgroup_config):
                # print(group_name)
                # SetName_to_DfCaseEval[SetName] = df_case_eval_by_group

                eval_instance = BanditPredEvalForOneEvalSet(
                    setname = setname,
                    df_case_eval = df_case_eval_by_group, 
                    action_name = action_name, 
                    reward_predicted_name = reward_predicted_name,
                    action_distribution_name = action_distribution_name,
                )
                eval_results = eval_instance.get_evaluation_report()
                report_list.append(eval_results)
        df_report_full = pd.DataFrame(report_list)
        columns_dfreport = [i for i in df_report_full.columns if 'df_' != i[:3]]
        df_report_neat = df_report_full[columns_dfreport].reset_index(drop = True)  
        self.df_report_full = df_report_full
        self.df_report_neat = df_report_neat






            


