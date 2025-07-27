import numpy as np 
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

    

# Part 1: 
# -------- df_case_eval
# -------- y_real_label_name
# -------- y_pred_score_name

# fn1

class BinaryPredEvalForOneEvalSet:

    def __init__(self, 
                 setname, 
                 df_case_eval, 
                 y_real_label_name, 
                 y_pred_score_name,
                 EachThreshold_step = 20, 
                 PredScoreGroup_step = 20, 
                 GroupNum = 10,
                 ):
        
        self.setname = setname
        self.df_case_eval = df_case_eval
        self.y_real_label_name = y_real_label_name
        self.y_pred_score_name = y_pred_score_name
        self.y_pred_score = df_case_eval[y_pred_score_name]
        self.y_real_label = df_case_eval[y_real_label_name]
        # self.y_read_score # not available
        # self.y_real_label # need to be determined by threshold

        self.EachThreshold_step = EachThreshold_step
        self.PredScoreGroup_step = PredScoreGroup_step
        self.GroupNum = GroupNum


    def get_evaluation_report(self):

        report = {'name': self.setname}
        d = self.get_evaluations()
        for k, v in d.items(): report[k] = v

        df_auc = self.get_df_auc()
        df_pr = self.get_df_pr()
        df_EachThreshold  = self.get_df_EachThreshold()
        df_PredScoreGroup = self.get_df_PredScoreGroup()
        df_TheNthGroup    = self.get_df_TheNthGroup()
        df_BtmNthGroup    = self.get_df_BtmNthGroup()
        df_TopNthGroup    = self.get_df_TopNthGroup()

        report['df_auc']            = df_auc.to_dict(orient = 'records')
        report['df_pr']             = df_pr # this is a dict
        report['df_EachThreshold']  = df_EachThreshold.to_dict(orient = 'records')
        report['df_PredScoreGroup'] = df_PredScoreGroup.to_dict(orient = 'records')
        report['df_TheNthGroup']    = df_TheNthGroup.to_dict(orient = 'records')
        report['df_BtmNthGroup']    = df_BtmNthGroup.to_dict(orient = 'records')
        report['df_TopNthGroup']    = df_TopNthGroup.to_dict(orient = 'records')
        report['sorted_pred_score'] = self.y_pred_score.sort_values(ascending = True).to_list()
        # here report is dictionary.
        return report


    # fn3
    def get_evaluations(self, 
                        y_real_label = None, 
                        y_pred_score = None
                        ):
        
        if y_real_label is None:
            y_real_label = self.y_real_label
        if y_pred_score is None:
            y_pred_score = self.y_pred_score
        
        SampleNum = len(y_real_label)
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
        
        RealPosNum = (np.array(y_real_label) == 1).sum()
        RealPosRate = round(RealPosNum / SampleNum, 4)
        if RealPosRate > 0 and RealPosRate < 1:
            fpr, tpr, thresholds = roc_curve(y_real_label, y_pred_score)
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = None
            
        d = {
            'SampleNum': SampleNum,
            'RealPosNum': RealPosNum, 
            'RealPosRate': RealPosRate,
            'roc_auc': roc_auc,
            'PredScoreMean': round(y_pred_score.mean(), 4),
            'PredScoreStd': round(y_pred_score.std(), 4),
            'PredScoreMin': round(y_pred_score.min(), 4),
            'PredScoreMax': round(y_pred_score.max(), 4),
        }

        RealPosNum = y_real_label.sum() - 1
        threshold_point = y_pred_score.sort_values(ascending = False).iloc[RealPosNum]
        threshold_point = round(threshold_point, 4)
        for threshold in [0.5, threshold_point]:
            results = self.get_evaluations_with_threshold(threshold)
            for k, v in results.items(): 
                key = f'0.5_{k}' if threshold == 0.5 else f'BLC_{k}'
                d[key] = v
        return d
        

    def get_evaluations_with_threshold(self,
                                       threshold, 
                                       y_real_label = None,
                                       y_pred_score = None):
        if y_real_label is None:
            y_real_label = self.y_real_label
        if y_pred_score is None:
            y_pred_score = self.y_pred_score
        y_pred_label = (y_pred_score >= threshold).astype(int)
        
        accuracy  = round(accuracy_score(y_real_label, y_pred_label), 4)
        SampleNum = len(y_real_label)
        PredPosNum = (np.array(y_pred_label) == 1).sum()
        RealPosNum = (np.array(y_real_label) == 1).sum()
    
        if PredPosNum > 0:
            precision = round(precision_score(y_real_label, y_pred_label), 4)
        else:
            precision = None
            
        if RealPosNum > 0:
            recall    = round(recall_score(y_real_label, y_pred_label), 4)
        else:
            recall = None
            
        if PredPosNum > 0 and RealPosNum > 0:
            f1 = f1_score(y_real_label, y_pred_label)
        else:
            f1 = None
        
        d = {
            'threshold': threshold,
            'accu': accuracy,
            'precise/PredPosAccu': precision,
            'recall /RealPosAccu': recall,
            'f1': f1,
            'SampleNum': SampleNum, 
            'PredPosNum': PredPosNum,
            'PredPosRate': round(PredPosNum / SampleNum, 4),
            'RealPosNum': RealPosNum,
            'RealPosRate': round(RealPosNum / SampleNum, 4),
        }
        return d


    def get_df_auc(self, 
                   y_real_label = None, 
                   y_pred_score = None):

        if y_real_label is None:
            y_real_label = self.y_real_label
        if y_pred_score is None:
            y_pred_score = self.y_pred_score

        fpr, tpr, thresholds = roc_curve(y_real_label, y_pred_score)
        df_auc = pd.DataFrame({
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
        })
        return df_auc
    

    def get_df_pr(self, 
                  y_real_label = None, 
                  y_pred_score = None):
        
        if y_real_label is None:
            y_real_label = self.y_real_label
        if y_pred_score is None:
            y_pred_score = self.y_pred_score


        precision, recall, thresholds = precision_recall_curve(y_real_label, y_pred_score)
        df_pr = {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
        }
        return df_pr 


    def get_df_EachThreshold(self, EachThreshold_step = None):
        
        if EachThreshold_step is None:
            EachThreshold_step = self.EachThreshold_step

        step = EachThreshold_step
        threshold_list = np.round(np.arange(0, 1 + step, step), 3)
        L = []
        for threshold in threshold_list:
            d = self.get_evaluations_with_threshold(threshold)
            L.append(d)
        df_EachThreshold = pd.DataFrame(L)
        return df_EachThreshold


    def get_df_PredScoreGroup(self, PredScoreGroup_step = None):
        y_pred_score = self.y_pred_score
        y_real_label = self.y_real_label
        if PredScoreGroup_step is None:
            PredScoreGroup_step = self.PredScoreGroup_step

        step = PredScoreGroup_step
        threshold_list = np.arange(0, 1, step)

        L = []
        for threshold in threshold_list:
            start = round(threshold,        3)
            end   = round(threshold + step, 3)
            index = (y_pred_score>= start) & (y_pred_score < end)

            # df_case = df_case_eval[]
            y_pred_score_subgroup = y_pred_score[index]
            y_real_label_subgroup = y_real_label[index]
            d = self.get_evaluations(y_real_label_subgroup, y_pred_score_subgroup)
            d['SubGroup'] = f'PredScore{start}-{end}'
            L.append(d)
        df_PredScoreGroup = pd.DataFrame(L) 
        return df_PredScoreGroup


    def get_df_TheNthGroup(self, GroupNum = None):

        if GroupNum is None: GroupNum = self.GroupNum
        
        df_case_eval = self.df_case_eval
        y_real_label_name = self.y_real_label_name
        y_pred_score_name = self.y_pred_score_name

        # topic 4: df_Nth_group_eval
        group_num = GroupNum
        df_case_eval = df_case_eval.sort_values(by = y_pred_score_name, ascending = True) 
        group_index_list = np.array_split(np.arange(len(df_case_eval)), group_num)
        group_size_list  = np.array([len(i) for i in group_index_list]).cumsum()
        
        L = []
        size = 100 / group_num
        for idx, end_index in enumerate(group_size_list):
            start = 0 if idx == 0 else group_size_list[idx - 1]
            end = end_index
            df_case = df_case_eval.iloc[start:end]
            y_real_label = df_case[y_real_label_name]
            y_pred_score = df_case[y_pred_score_name]
            d = self.get_evaluations(y_real_label, y_pred_score)
            s, e = round(size * idx, 2), round(size * (idx + 1), 2)
            d['Group'] = f'{s}%-{e}%'
            L.append(d)
        df_TheNthGroup = pd.DataFrame(L)
        return df_TheNthGroup
    

    def get_df_BtmNthGroup(self, GroupNum = None):

        if GroupNum is None: GroupNum = self.GroupNum
        
        df_case_eval = self.df_case_eval
        y_real_label_name = self.y_real_label_name
        y_pred_score_name = self.y_pred_score_name

        # topic 5: df_Btm_group_eval 
        group_num = GroupNum
        df_case_eval = df_case_eval.sort_values(by = y_pred_score_name, ascending = True) 
        group_index_list = np.array_split(np.arange(len(df_case_eval)), group_num)
        group_size_list = np.array([len(i) for i in group_index_list]).cumsum()
        
        L = []
        size = 100 / group_num
        for idx, end_index in enumerate(group_size_list):
            start = 0 # if idx == 0 else group_size_list[idx - 1]
            end = end_index
            df_case = df_case_eval.iloc[start:end]
            y_real_label = df_case[y_real_label_name]
            y_pred_score = df_case[y_pred_score_name]
            d = self.get_evaluations(y_real_label, y_pred_score)
            s, e = round(size * idx, 2), round(size * (idx + 1), 2)
            d['Group'] = f'Btm{e}%'
            L.append(d)
        df_BtmNthGroup = pd.DataFrame(L)
        return df_BtmNthGroup


    def get_df_TopNthGroup(self, GroupNum = None):
        if GroupNum is None: GroupNum = self.GroupNum
        
        df_case_eval = self.df_case_eval
        y_real_label_name = self.y_real_label_name
        y_pred_score_name = self.y_pred_score_name

        # topic 6: df_Top_group_eval 
        group_num = GroupNum
        df_case_eval = df_case_eval.sort_values(by = y_pred_score_name, ascending = False) 
        group_index_list = np.array_split(np.arange(len(df_case_eval)), group_num)
        group_size_list = np.array(list(reversed([len(i) for i in group_index_list]))).cumsum()
        
        L = []
        size = 100 / group_num
        for idx, end_index in enumerate(group_size_list):
            start = 0 # if idx == 0 else group_size_list[idx - 1]
            end = end_index
            df_case = df_case_eval.iloc[start:end]
            y_real_label = df_case[y_real_label_name]
            y_pred_score = df_case[y_pred_score_name]
            d = self.get_evaluations(y_real_label, y_pred_score)
            # d = evaluate_df_case_by_pred_score(df_case, y_real_label_name, y_pred_score_name)
            s, e = round(size * idx, 2), round(size * (idx + 1), 2)
            d['Group'] = f'Top{e}%'
            L.append(d)
        df_TopNthGroup = pd.DataFrame(L)
        return df_TopNthGroup


class BinaryPredEvalPlot:
    def plot_group_eval(df, rate_cols, number_cols, group_col):
        
        # Create figure and subplots
        fig = plt.figure(figsize=(18, 12), dpi=200)  # Set a large figure size and high DPI
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], figure=fig)

        # Upper plot for rate columns
        ax1 = fig.add_subplot(gs[0])
        for col in rate_cols:
            ax1.plot(df[group_col], df[col], marker='o', label=col)
            
        # Disable x-tick labels on the upper plot using tick_params
        ax1.tick_params(axis='x',          # Changes apply to the x-axis
                        which='both',      # Both major and minor ticks are affected
                        bottom=False,      # Ticks along the bottom edge are off
                        top=False,         # Ticks along the top edge are off
                        labelbottom=False) # Labels along the bottom edge are off

        ax1.legend(loc='best')
        ax1.set_title('Rates by Group')
        ax1.grid(True)

        # Lower plot for number columns
        # Lower plot for number columns
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        len_numcols = len(number_cols)
        width = 1 / (len_numcols + 1)  # Width of the bars
        ind = np.arange(len(df))  # the x locations for the groups

        # Calculate the correction to center the bars around the tick marks
        # print(len(number_cols))
        correction = 0 if len(number_cols) % 2 != 0 else width / 2

        # Loop to plot each bar
        for i, col in enumerate(number_cols):
            # Compute the position for each bar
            position = ind - correction + ((i - int(len(number_cols) / 2)) * width)
            # print(i, position)
            ax2.bar(position, df[col], width, label=col)
            
        ax2.set_xticks(ind)
        ax2.set_xticklabels(df[group_col], rotation=90)  # Rotate x-tick labels for readability
        ax2.legend(loc='best')
        ax2.set_title('Numbers by Group')

        plt.subplots_adjust(hspace=0.1)  # Adjust space between subplots
        plt.show()


    # Plot ROC curve

    def plot_roc_curve(result_auc):
        df_auc = result_auc['df_auc']
        fpr = df_auc['fpr']
        tpr = df_auc['tpr']
        thresholds = df_auc['thresholds']
        roc_auc = result_auc['auc']
        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        # Select thresholds to annotate on the curve
        # You can choose specific thresholds that are of interest or select them based on criteria
        indices_to_annotate = [len(thresholds) // 3 * 1, 
                            len(thresholds) // 2, 
                            len(thresholds) // 3 * 2, 
                            len(thresholds) - 1]  # Example: start, middle, end
        for i in indices_to_annotate:
            plt.annotate(f'Threshold={thresholds[i]:.2f}', (fpr[i], tpr[i]),
                        textcoords="offset points", xytext=(-10,-10), ha='center')

        plt.show()
        
    # Plot ROC curve

    def plot_roc_curve(result_pr):
        df_pr = result_pr['df_pr']
        precision = df_pr['precision']
        recall  = df_pr['recall']
        thresholds = df_pr['thresholds']
        plt.figure(figsize=(10, 10))
        plt.plot(recall, precision, label='Precision-Recall curve')
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlim([0.0, 1.05])
        # plt.ylim([0.0, 1.05])
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        # plt.show()

        # Select thresholds to annotate on the curve
        # You can choose specific thresholds that are of interest or select them based on criteria
        indices_to_annotate = [len(thresholds) // 3 * 1, 
                            len(thresholds) // 2, 
                            len(thresholds) // 3 * 2, 
                            
                            len(thresholds) // 4 * 3, 
                            len(thresholds) - 1]  # Example: start, middle, end
        for i in indices_to_annotate:
            plt.annotate(f'Threshold={thresholds[i]:.2f}', 
                        (recall[i], precision[i]),
                        textcoords="offset points", 
                        xytext=(-10,-10), 
                        ha='center')

        plt.show()




class BinaryPredEval(BinaryPredEvalPlot):

    def __init__(self, 
                 df_case_eval, 
                 subgroup_config_list, 
                 y_real_label_name, 
                 y_pred_score_name,
                 EachThreshold_step = 20, 
                 PredScoreGroup_step = 20, 
                 GroupNum = 10,):
        
        # self.SetName_to_DfCaseEval = SetName_to_DfCaseEval
        # for setname, df_case_eval in SetName_to_DfCaseEval.items():
        #     df_case_eval['SetName'] = setname


        report_list = []
        for subgroup_config in subgroup_config_list:
            for setname, df_case_eval_by_group in df_case_eval.groupby(subgroup_config):
                # print(group_name)
                # SetName_to_DfCaseEval[SetName] = df_case_eval_by_group

                eval_instance = BinaryPredEvalForOneEvalSet(
                    setname = setname,
                    df_case_eval = df_case_eval_by_group, 
                    y_real_label_name = y_real_label_name,
                    y_pred_score_name = y_pred_score_name,
                    EachThreshold_step = EachThreshold_step,
                    PredScoreGroup_step = PredScoreGroup_step,
                    GroupNum = GroupNum,
                )
                eval_results = eval_instance.get_evaluation_report()
                report_list.append(eval_results)

        df_report_full = pd.DataFrame(report_list)
        columns_dfreport = [i for i in df_report_full.columns if 'df_' != i[:3]]
        df_report_neat = df_report_full[columns_dfreport].reset_index(drop = True)  
        self.df_report_full = df_report_full
        self.df_report_neat = df_report_neat






            


