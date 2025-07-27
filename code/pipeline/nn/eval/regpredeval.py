import numpy as np 
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import seaborn as sns

class RegressionPredEvalForOneEvalSet:

    def __init__(self, 
                 setname, 
                 df_case_eval, 
                 y_real_value_name, 
                 y_pred_value_name,
                 ResidualGroup_step = 20, 
                 PredValueGroup_step = 20, 
                 GroupNum = 10,
                 ):
        
        self.setname = setname
        self.df_case_eval = df_case_eval
        self.y_real_value_name = y_real_value_name
        self.y_pred_value_name = y_pred_value_name
        self.y_pred_value = df_case_eval[y_pred_value_name]
        self.y_real_value = df_case_eval[y_real_value_name]

        self.ResidualGroup_step = ResidualGroup_step
        self.PredValueGroup_step = PredValueGroup_step
        self.GroupNum = GroupNum


    def get_evaluation_report(self):

        report = {'name': self.setname}
        d = self.get_evaluations()
        for k, v in d.items(): report[k] = v

        df_scatter = self.get_df_scatter()
        df_residual = self.get_df_residual()
        
        # Only calculate if ResidualGroup_step is not None
        if self.ResidualGroup_step is not None:
            df_ResidualGroup = self.get_df_ResidualGroup()
            report['df_ResidualGroup'] = df_ResidualGroup.to_dict(orient = 'records')
        
        # Only calculate if PredValueGroup_step is not None
        if self.PredValueGroup_step is not None:
            df_PredValueGroup = self.get_df_PredValueGroup()
            report['df_PredValueGroup'] = df_PredValueGroup.to_dict(orient = 'records')
        
        # Only calculate if GroupNum is not None
        if self.GroupNum is not None:
            df_TheNthGroup = self.get_df_TheNthGroup()
            df_BtmNthGroup = self.get_df_BtmNthGroup()
            df_TopNthGroup = self.get_df_TopNthGroup()
            report['df_TheNthGroup'] = df_TheNthGroup.to_dict(orient = 'records')
            report['df_BtmNthGroup'] = df_BtmNthGroup.to_dict(orient = 'records')
            report['df_TopNthGroup'] = df_TopNthGroup.to_dict(orient = 'records')

        report['df_scatter'] = df_scatter.to_dict(orient = 'records')
        report['df_residual'] = df_residual.to_dict(orient = 'records')
        report['sorted_pred_value'] = self.y_pred_value.sort_values(ascending = True).to_list()
        report['sorted_real_value'] = self.y_real_value.sort_values(ascending = True).to_list()
        # here report is dictionary.
        return report


    def get_evaluations(self, 
                        y_real_value = None, 
                        y_pred_value = None
                        ):
        
        if y_real_value is None:
            y_real_value = self.y_real_value
        if y_pred_value is None:
            y_pred_value = self.y_pred_value
        
        SampleNum = len(y_real_value)
        if SampleNum == 0: 
            d = {
                'SampleNum': 0,
                'RealValueMean': None,
                'RealValueStd': None,
                'RealValueMin': None, 
                'RealValueMax': None,
                'PredValueMean': None,
                'PredValueStd': None,
                'PredValueMin': None, 
                'PredValueMax': None,
                'mse': None,
                'rmse': None,
                'mae': None,
                'mape': None,
                'r2': None,
                'correlation': None,
                'mean_residual': None,
                'std_residual': None,
            }
            return d
        
        # Calculate basic statistics
        real_mean = round(y_real_value.mean(), 4)
        real_std = round(y_real_value.std(), 4)
        real_min = round(y_real_value.min(), 4)
        real_max = round(y_real_value.max(), 4)
        
        pred_mean = round(y_pred_value.mean(), 4)
        pred_std = round(y_pred_value.std(), 4)
        pred_min = round(y_pred_value.min(), 4)
        pred_max = round(y_pred_value.max(), 4)
        
        # Calculate regression metrics
        mse = round(mean_squared_error(y_real_value, y_pred_value), 4)
        rmse = round(np.sqrt(mse), 4)
        mae = round(mean_absolute_error(y_real_value, y_pred_value), 4)
        
        # Calculate MAPE only if no zero values in y_real_value
        if (y_real_value == 0).any():
            mape = None
        else:
            mape = round(mean_absolute_percentage_error(y_real_value, y_pred_value), 4)
        
        r2 = round(r2_score(y_real_value, y_pred_value), 4)
        
        # Calculate correlation
        correlation = round(np.corrcoef(y_real_value, y_pred_value)[0, 1], 4)
        
        # Calculate residuals
        residuals = y_pred_value - y_real_value
        mean_residual = round(residuals.mean(), 4)
        std_residual = round(residuals.std(), 4)
            
        d = {
            'SampleNum': SampleNum,
            'RealValueMean': real_mean,
            'RealValueStd': real_std,
            'RealValueMin': real_min, 
            'RealValueMax': real_max,
            'PredValueMean': pred_mean,
            'PredValueStd': pred_std,
            'PredValueMin': pred_min, 
            'PredValueMax': pred_max,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'correlation': correlation,
            'mean_residual': mean_residual,
            'std_residual': std_residual,
        }
        return d


    def get_df_scatter(self, 
                       y_real_value = None, 
                       y_pred_value = None):

        if y_real_value is None:
            y_real_value = self.y_real_value
        if y_pred_value is None:
            y_pred_value = self.y_pred_value

        df_scatter = pd.DataFrame({
            'y_real': y_real_value,
            'y_pred': y_pred_value,
            'residual': y_pred_value - y_real_value,
        })
        return df_scatter
    

    def get_df_residual(self, 
                        y_real_value = None, 
                        y_pred_value = None):
        
        if y_real_value is None:
            y_real_value = self.y_real_value
        if y_pred_value is None:
            y_pred_value = self.y_pred_value

        residuals = y_pred_value - y_real_value
        df_residual = pd.DataFrame({
            'y_real': y_real_value,
            'y_pred': y_pred_value,
            'residual': residuals,
            'abs_residual': np.abs(residuals),
            'squared_residual': residuals ** 2,
        })
        return df_residual


    def get_df_ResidualGroup(self, ResidualGroup_step = None):
        if ResidualGroup_step is None:
            ResidualGroup_step = self.ResidualGroup_step
        
        # If ResidualGroup_step is still None, return empty DataFrame
        if ResidualGroup_step is None:
            return pd.DataFrame()
            
        y_pred_value = self.y_pred_value
        y_real_value = self.y_real_value

        residuals = y_pred_value - y_real_value
        residual_min = residuals.min()
        residual_max = residuals.max()
        residual_range = residual_max - residual_min
        
        if residual_range == 0:
            # All residuals are the same
            L = []
            d = self.get_evaluations(y_real_value, y_pred_value)
            d['SubGroup'] = f'Residual{residual_min:.3f}-{residual_max:.3f}'
            L.append(d)
            df_ResidualGroup = pd.DataFrame(L)
            return df_ResidualGroup

        step_size = residual_range / (100 / ResidualGroup_step)
        threshold_list = np.arange(residual_min, residual_max + step_size, step_size)

        L = []
        for i in range(len(threshold_list) - 1):
            start = round(threshold_list[i], 3)
            end = round(threshold_list[i + 1], 3)
            index = (residuals >= start) & (residuals < end)
            
            if i == len(threshold_list) - 2:  # Last group includes the maximum
                index = (residuals >= start) & (residuals <= end)

            y_pred_value_subgroup = y_pred_value[index]
            y_real_value_subgroup = y_real_value[index]
            
            if len(y_pred_value_subgroup) > 0:
                d = self.get_evaluations(y_real_value_subgroup, y_pred_value_subgroup)
                d['SubGroup'] = f'Residual{start}-{end}'
                L.append(d)
                
        df_ResidualGroup = pd.DataFrame(L) 
        return df_ResidualGroup


    def get_df_PredValueGroup(self, PredValueGroup_step = None):
        if PredValueGroup_step is None:
            PredValueGroup_step = self.PredValueGroup_step
        
        # If PredValueGroup_step is still None, return empty DataFrame
        if PredValueGroup_step is None:
            return pd.DataFrame()
            
        y_pred_value = self.y_pred_value
        y_real_value = self.y_real_value

        pred_min = y_pred_value.min()
        pred_max = y_pred_value.max()
        pred_range = pred_max - pred_min
        
        if pred_range == 0:
            # All predictions are the same
            L = []
            d = self.get_evaluations(y_real_value, y_pred_value)
            d['SubGroup'] = f'PredValue{pred_min:.3f}-{pred_max:.3f}'
            L.append(d)
            df_PredValueGroup = pd.DataFrame(L)
            return df_PredValueGroup

        step_size = pred_range / (100 / PredValueGroup_step)
        threshold_list = np.arange(pred_min, pred_max + step_size, step_size)

        L = []
        for i in range(len(threshold_list) - 1):
            start = round(threshold_list[i], 3)
            end = round(threshold_list[i + 1], 3)
            index = (y_pred_value >= start) & (y_pred_value < end)
            
            if i == len(threshold_list) - 2:  # Last group includes the maximum
                index = (y_pred_value >= start) & (y_pred_value <= end)

            y_pred_value_subgroup = y_pred_value[index]
            y_real_value_subgroup = y_real_value[index]
            
            if len(y_pred_value_subgroup) > 0:
                d = self.get_evaluations(y_real_value_subgroup, y_pred_value_subgroup)
                d['SubGroup'] = f'PredValue{start}-{end}'
                L.append(d)
                
        df_PredValueGroup = pd.DataFrame(L) 
        return df_PredValueGroup


    def get_df_TheNthGroup(self, GroupNum = None):
        if GroupNum is None: 
            GroupNum = self.GroupNum
        
        # If GroupNum is still None, return empty DataFrame
        if GroupNum is None:
            return pd.DataFrame()
        
        df_case_eval = self.df_case_eval
        y_real_value_name = self.y_real_value_name
        y_pred_value_name = self.y_pred_value_name

        # Sort by predicted values
        group_num = GroupNum
        df_case_eval = df_case_eval.sort_values(by = y_pred_value_name, ascending = True) 
        group_index_list = np.array_split(np.arange(len(df_case_eval)), group_num)
        group_size_list  = np.array([len(i) for i in group_index_list]).cumsum()
        
        L = []
        size = 100 / group_num
        for idx, end_index in enumerate(group_size_list):
            start = 0 if idx == 0 else group_size_list[idx - 1]
            end = end_index
            df_case = df_case_eval.iloc[start:end]
            y_real_value = df_case[y_real_value_name]
            y_pred_value = df_case[y_pred_value_name]
            d = self.get_evaluations(y_real_value, y_pred_value)
            s, e = round(size * idx, 2), round(size * (idx + 1), 2)
            d['Group'] = f'{s}%-{e}%'
            L.append(d)
        df_TheNthGroup = pd.DataFrame(L)
        return df_TheNthGroup
    

    def get_df_BtmNthGroup(self, GroupNum = None):
        if GroupNum is None: 
            GroupNum = self.GroupNum
        
        # If GroupNum is still None, return empty DataFrame
        if GroupNum is None:
            return pd.DataFrame()
        
        df_case_eval = self.df_case_eval
        y_real_value_name = self.y_real_value_name
        y_pred_value_name = self.y_pred_value_name

        # Sort by predicted values (ascending)
        group_num = GroupNum
        df_case_eval = df_case_eval.sort_values(by = y_pred_value_name, ascending = True) 
        group_index_list = np.array_split(np.arange(len(df_case_eval)), group_num)
        group_size_list = np.array([len(i) for i in group_index_list]).cumsum()
        
        L = []
        size = 100 / group_num
        for idx, end_index in enumerate(group_size_list):
            start = 0  # Always start from bottom
            end = end_index
            df_case = df_case_eval.iloc[start:end]
            y_real_value = df_case[y_real_value_name]
            y_pred_value = df_case[y_pred_value_name]
            d = self.get_evaluations(y_real_value, y_pred_value)
            s, e = round(size * idx, 2), round(size * (idx + 1), 2)
            d['Group'] = f'Btm{e}%'
            L.append(d)
        df_BtmNthGroup = pd.DataFrame(L)
        return df_BtmNthGroup


    def get_df_TopNthGroup(self, GroupNum = None):
        if GroupNum is None: 
            GroupNum = self.GroupNum
        
        # If GroupNum is still None, return empty DataFrame
        if GroupNum is None:
            return pd.DataFrame()
        
        df_case_eval = self.df_case_eval
        y_real_value_name = self.y_real_value_name
        y_pred_value_name = self.y_pred_value_name

        # Sort by predicted values (descending for top groups)
        group_num = GroupNum
        df_case_eval = df_case_eval.sort_values(by = y_pred_value_name, ascending = False) 
        group_index_list = np.array_split(np.arange(len(df_case_eval)), group_num)
        group_size_list = np.array(list(reversed([len(i) for i in group_index_list]))).cumsum()
        
        L = []
        size = 100 / group_num
        for idx, end_index in enumerate(group_size_list):
            start = 0  # Always start from top
            end = end_index
            df_case = df_case_eval.iloc[start:end]
            y_real_value = df_case[y_real_value_name]
            y_pred_value = df_case[y_pred_value_name]
            d = self.get_evaluations(y_real_value, y_pred_value)
            s, e = round(size * idx, 2), round(size * (idx + 1), 2)
            d['Group'] = f'Top{e}%'
            L.append(d)
        df_TopNthGroup = pd.DataFrame(L)
        return df_TopNthGroup


class RegressionPredEvalPlot:
    
    def plot_group_eval(df, rate_cols, number_cols, group_col):
        
        # Create figure and subplots
        fig = plt.figure(figsize=(18, 12), dpi=200)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], figure=fig)

        # Upper plot for rate columns
        ax1 = fig.add_subplot(gs[0])
        for col in rate_cols:
            ax1.plot(df[group_col], df[col], marker='o', label=col)
            
        # Disable x-tick labels on the upper plot
        ax1.tick_params(axis='x',
                        which='both',
                        bottom=False,
                        top=False,
                        labelbottom=False)

        ax1.legend(loc='best')
        ax1.set_title('Metrics by Group')
        ax1.grid(True)

        # Lower plot for number columns
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        len_numcols = len(number_cols)
        width = 1 / (len_numcols + 1)
        ind = np.arange(len(df))

        correction = 0 if len(number_cols) % 2 != 0 else width / 2

        for i, col in enumerate(number_cols):
            position = ind - correction + ((i - int(len(number_cols) / 2)) * width)
            ax2.bar(position, df[col], width, label=col)
            
        ax2.set_xticks(ind)
        ax2.set_xticklabels(df[group_col], rotation=90)
        ax2.legend(loc='best')
        ax2.set_title('Sample Numbers by Group')

        plt.subplots_adjust(hspace=0.1)
        plt.show()


    def plot_scatter(result_scatter, setname=''):
        df_scatter = pd.DataFrame(result_scatter['df_scatter'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Predicted vs Actual scatter plot
        axes[0, 0].scatter(df_scatter['y_real'], df_scatter['y_pred'], alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(df_scatter['y_real'].min(), df_scatter['y_pred'].min())
        max_val = max(df_scatter['y_real'].max(), df_scatter['y_pred'].max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title(f'Predicted vs Actual Values - {setname}')
        axes[0, 0].grid(True)
        
        # Residuals vs Predicted
        axes[0, 1].scatter(df_scatter['y_pred'], df_scatter['residual'], alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title(f'Residuals vs Predicted - {setname}')
        axes[0, 1].grid(True)
        
        # Residuals vs Actual
        axes[1, 0].scatter(df_scatter['y_real'], df_scatter['residual'], alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Actual Values')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title(f'Residuals vs Actual - {setname}')
        axes[1, 0].grid(True)
        
        # Histogram of residuals
        axes[1, 1].hist(df_scatter['residual'], bins=30, alpha=0.7, density=True)
        axes[1, 1].axvline(x=0, color='r', linestyle='--')
        
        # Add normal distribution overlay
        mu, sigma = df_scatter['residual'].mean(), df_scatter['residual'].std()
        x = np.linspace(df_scatter['residual'].min(), df_scatter['residual'].max(), 100)
        axes[1, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal fit')
        
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title(f'Distribution of Residuals - {setname}')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()


    def plot_residual_analysis(result_residual, setname=''):
        df_residual = pd.DataFrame(result_residual['df_residual'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Q-Q plot for normality check
        stats.probplot(df_residual['residual'], dist="norm", plot=axes[0, 0])
        axes[0, 0].set_title(f'Q-Q Plot of Residuals - {setname}')
        axes[0, 0].grid(True)
        
        # Scale-Location plot (sqrt of absolute residuals vs predicted)
        sqrt_abs_resid = np.sqrt(np.abs(df_residual['residual']))
        axes[0, 1].scatter(df_residual['y_pred'], sqrt_abs_resid, alpha=0.6)
        
        # Add trend line
        z = np.polyfit(df_residual['y_pred'], sqrt_abs_resid, 1)
        p = np.poly1d(z)
        axes[0, 1].plot(df_residual['y_pred'].sort_values(), 
                       p(df_residual['y_pred'].sort_values()), "r--", alpha=0.8)
        
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('√|Residuals|')
        axes[0, 1].set_title(f'Scale-Location Plot - {setname}')
        axes[0, 1].grid(True)
        
        # Box plot of residuals
        axes[1, 0].boxplot(df_residual['residual'])
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title(f'Box Plot of Residuals - {setname}')
        axes[1, 0].grid(True)
        
        # Residuals vs Index (order)
        axes[1, 1].scatter(range(len(df_residual)), df_residual['residual'], alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Observation Index')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title(f'Residuals vs Order - {setname}')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()


class RegressionPredEval(RegressionPredEvalPlot):

    def __init__(self, 
                 df_case_eval, 
                 subgroup_config_list, 
                 y_real_value_name, 
                 y_pred_value_name,
                 ResidualGroup_step = 20, 
                 PredValueGroup_step = 20, 
                 GroupNum = 10,):
        
        report_list = []
        for subgroup_config in subgroup_config_list:
            for setname, df_case_eval_by_group in df_case_eval.groupby(subgroup_config):
                
                eval_instance = RegressionPredEvalForOneEvalSet(
                    setname = setname,
                    df_case_eval = df_case_eval_by_group, 
                    y_real_value_name = y_real_value_name,
                    y_pred_value_name = y_pred_value_name,
                    ResidualGroup_step = ResidualGroup_step,
                    PredValueGroup_step = PredValueGroup_step,
                    GroupNum = GroupNum,
                )
                eval_results = eval_instance.get_evaluation_report()
                report_list.append(eval_results)

        df_report_full = pd.DataFrame(report_list)
        columns_dfreport = [i for i in df_report_full.columns if 'df_' != i[:3]]
        df_report_neat = df_report_full[columns_dfreport].reset_index(drop = True)  
        self.df_report_full = df_report_full
        self.df_report_neat = df_report_neat