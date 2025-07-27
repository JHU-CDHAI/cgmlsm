"""
Helper functions for seqeval:
All matrix evaluation function are defined here and should be called by SeqEvalForOneDataPoint.get_metric_scores()
Format is :
func(y_real_seq, y_pred_seq)
(should convert / assume both sequence to array to promote faster prediction)

"""
import numpy as np
import pandas as pd
from scipy.stats import entropy, gaussian_kde, entropy as kl_divergence
from scipy.spatial.distance import directed_hausdorff
from sklearn.mixture import GaussianMixture
from statsmodels.tsa.seasonal import STL
from numpy.polynomial.polynomial import Polynomial
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings("ignore")
rounding_coef = 5
def fit_polynomial(y, degree=3):
    x = np.arange(len(y))
    coefs = np.polyfit(x, y, degree)
    poly = np.poly1d(coefs)
    y_fit = poly(x)
    residuals = y - y_fit
    return {'poly_coeffs': np.round(coefs, rounding_coef), 'poly_residual_std': np.round(np.std(residuals), rounding_coef)}

def fit_gmm(y, n_components=2):
    y = np.array(y).reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(y)
    return {
        'gmm_means': np.round(gmm.means_.flatten(), rounding_coef),
        'gmm_stds': np.round(np.sqrt(gmm.covariances_).flatten(), rounding_coef),
        'gmm_weights': np.round(gmm.weights_.flatten(), rounding_coef)
    }

def stl_decomposition(y, period=12):
    series = pd.Series(y)
    result = STL(series, period=period).fit()
    return {
        'stl_trend_std': np.round(np.std(result.trend), rounding_coef),
        'stl_seasonal_std': np.round(np.std(result.seasonal), rounding_coef),
        'stl_resid_std': np.round(np.std(result.resid), rounding_coef)
    }

def autocorr_features(y, nlags=10):
    y = np.array(y)
    autocorrs = [np.corrcoef(y[:-lag], y[lag:])[0, 1] if lag < len(y) else 0 for lag in range(1, nlags+1)]
    return {f'autocorr_lag{lag+1}': ac for lag, ac in enumerate(autocorrs)}

def calc_entropy(y, bins=10):
    hist, bin_edges = np.histogram(y, bins=bins, density=True)
    hist = hist[hist > 0]
    return {'entropy': entropy(hist)}

def calc_frechet_distance(x, y):
    return {'frechet_distance': directed_hausdorff(np.stack((x, y), axis=1), np.stack((x, y), axis=1))[0]}

def calc_kl_divergence(p, q, bins=10):
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)
    p_hist += 1e-10
    q_hist += 1e-10
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return {'kl_divergence': kl_divergence(p_hist, q_hist)}

# Return all available metric function names
metrics_dict = {
    'polynomial': fit_polynomial,
    'gmm': fit_gmm,
    'stl': stl_decomposition,
    'autocorr': autocorr_features,
    'entropy': calc_entropy,
    'frechet': calc_frechet_distance,
    'kl': calc_kl_divergence
}



class SeqEvalHelper:
    def __init__(self, x_obs_seq, y_real_seq, y_pred_seq, metric_list=None):
        self.x_obs_seq  = np.array(x_obs_seq)
        self.y_real_seq = np.array(y_real_seq)
        self.y_pred_seq = np.array(y_pred_seq)
        self.metric_list = metric_list if metric_list else []
        self.metric_results = {}

    def get_metric_scores(self, metric_name):
        """
        Compute a single metric by name and return its results as a dict.
        """
        if metric_name not in metrics_dict:
            raise ValueError(f"Metric '{metric_name}' not recognized.")
        
        metric_func = metrics_dict[metric_name]

        # For metrics comparing prediction vs real (like frechet and KL)
        if metric_name in ['frechet', 'kl']:
            return metric_func(self.y_real_seq, self.y_pred_seq)
        
        # For metrics that only use the real sequence
        elif metric_name == 'polynomial':
            return metric_func(self.x_obs_seq, degree=3)
        elif metric_name == 'gmm':
            return metric_func(self.x_obs_seq, n_components=2)
        elif metric_name == 'stl':
            return metric_func(self.x_obs_seq, period=12)
        elif metric_name == 'autocorr':
            return metric_func(self.x_obs_seq, nlags=5)
        elif metric_name == 'entropy':
            return metric_func(self.x_obs_seq)
    def get_metric_scores_func(self, metric_name):
        """
        Return the metric function for a given name.
        Also computes and stores the result in self.metric_results.
        """
        if metric_name not in metrics_dict:
            raise ValueError(f"Metric '{metric_name}' not recognized.")
        
        metric_func = metrics_dict[metric_name]

        # Define a wrapped function that also saves the result to the instance
        def wrapped(inseq, oseq, seq = 'output'):
            if seq == 'input':
                eval_seq = self.x_obs_seq
            elif seq == 'output':
                eval_seq = self.y_pred_seq
            elif seq == 'real':
                eval_seq = self.y_real_seq
            else:
                raise ValueError(f"Invalid sequence type: {seq}")
            
            if metric_name in ['frechet', 'kl']:
                result = metric_func(self.y_real_seq, self.y_pred_seq)
            elif metric_name == 'polynomial':
                result = metric_func(eval_seq, degree=3)
            elif metric_name == 'gmm':
                result = metric_func(eval_seq, n_components=2)
            elif metric_name == 'stl':
                result = metric_func(eval_seq, period=12)
            elif metric_name == 'autocorr':
                result = metric_func(eval_seq, nlags=5)
            elif metric_name == 'entropy':
                result = metric_func(eval_seq)
            else:
                raise NotImplementedError(f"No logic defined for {metric_name}")
            
            self.metric_results[metric_name] = result
            return result

        return wrapped
        

