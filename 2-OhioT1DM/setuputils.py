import sys
import os
import logging
import pandas as pd
import datasets
from datasets import disable_caching; disable_caching()
from pprint import pprint
KEY = '2-NOTEBOOK'
WORKSPACE_PATH = os.getcwd().split(KEY)[0]
print(WORKSPACE_PATH); os.chdir(WORKSPACE_PATH)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s:%(asctime)s:(%(filename)s@%(lineno)d %(name)s)]: %(message)s')

# Suppress Python Deprecation Warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



# SPACE = {
#     'DATA_RAW': f'_Data/0-Data_Raw',
#     'DATA_RFT': f'_Data/1-Data_RFT',
#     'DATA_CASE': f'_Data/2-Data_CASE',
#     'DATA_AIDATA': f'_Data/3-Data_AIDATA',
#     'DATA_EXTERNAL': f'code/external',
#     'DATA_HFDATA': f'_Data/5-Data_HFData',
#     'CODE_FN': f'code/pipeline',
#     'MODEL_ROOT': f'./_Model',
# }
# assert os.path.exists(SPACE['CODE_FN']), f'{SPACE["CODE_FN"]} not found'
# print(SPACE['CODE_FN'])
# sys.path.append(SPACE['CODE_FN'])

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

#Get Data
# HFDataName = 'FairGlucoBench-Dart-v0518'
# path = os.path.join(SPACE['DATA_HFDATA'], HFDataName)

import os
import joblib
import pickle

# def load_timeseries_splits(save_root, splits=('train', 'valid', 'testid', 'testod')):
#     timeseries_ds = {}
#     for split in splits:
#         split_path = os.path.join(save_root, f"{split}_timeseries.pkl")
#         if not os.path.exists(split_path):
#             raise FileNotFoundError(f"Split file not found: {split_path}")
#         with open(split_path, 'rb') as f:
#             timeseries_ds[split] = pickle.load(f)
#         print(f"✅ Loaded {split}: {len(timeseries_ds[split])} series")
#     return timeseries_ds

# timeseries_ds = load_timeseries_splits(path)
# for k, v in timeseries_ds.items():
#     print(k, len(v['full_series_list']))

from datetime import datetime

def get_YYDDMMHH():
    # Get current time
    now = datetime.now()
    
    # Get year (last 2 digits)
    year = str(now.year)[-2:]
    
    # Get day of year (1-366)
    day_of_year = str(now.timetuple().tm_yday).zfill(3)
    
    # Get month (01-12)
    month = str(now.month).zfill(2)
    
    # Get hour (00-23)
    hour = str(now.hour).zfill(2)
    
    # Combine all components
    time_str = f"{year}{day_of_year}{month}{hour}"
    return time_str



#Plot Utils
import matplotlib.pyplot as plt
import os

#placeholder for scaler
class IdentityScaler:
    def fit(self, series):
        return self

    def transform(self, series):
        return series

    def inverse_transform(self, series):
        return series
    
def plot_prediction_for_idx(
    split: str,
    idx: int,
    model,
    timeseries_ds: dict,
    save_dir: str,
    input_length: int,
    output_length: int,
    model_type : str,
    COLUMNS_TO_USE = ['unique_id', 'ds', 'y']
):
    """
    Predicts and plots input vs. predicted vs. true series for a given idx and split.
    
    Parameters:
    - split: str, one of ['train', 'valid', 'testid', 'testod']
    - idx: int, index of the time series sample
    - model: a fitted Darts model
    - timeseries_ds: dict, loaded time series dataset
    - save_dir: str, directory to save the plot

    Saves:
    - PNG plot named "prediction_{model_name}_idx{idx}.png" in save_dir
    """

    # Extract required values
    # context_length = model.input_chunk_length
    prediction_length = 96#model.output_chunk_length
    output_length = output_length # prediction length and output length are not the same.
    model_name = model_type

    # Get data
    sample_df = timeseries_ds[split]
    sample_df = sample_df[sample_df['unique_id'] == idx]
    full_series = sample_df[COLUMNS_TO_USE]
    full_series_scaled = sample_df[COLUMNS_TO_USE]
    # scaler_list = sample_df['scaler_list']
    scaler = IdentityScaler() #scaler_list[idx]

    # Prepare input
    input_series = full_series.iloc[288-input_length:289]
    input_series_scaled = full_series_scaled.iloc[288-input_length:289]

    # Predict and inverse transform
    pred_series_scaled = model.predict(input_series_scaled)
    pred_series = pred_series_scaled['']#scaler.inverse_transform(pred_series_scaled)
    real_series = full_series[289:289+prediction_length] # plot all the prediction length

    # Plot
    plt.figure(figsize=(10, 4))
    full_series[288-input_length:].plot(label="Full", alpha=0.5, linewidth=4)
    input_series[:289].plot(label="Input")
    pred_series.plot(label="Predicted")
    real_series.plot(label="True")
    
    
    plt.axvline(x=full_series.time_index[288-input_length], color='black', linestyle='--', label=f"Input start")
    plt.axvline(x=full_series.time_index[289], color='black', linestyle='--', label=f"Prediction start")
    plt.axvline(x=full_series.time_index[288+output_length], color='black', linestyle='--', label=f"Prediction end")
    plt.legend()
    plt.title(f"{model_name} Prediction @ idx={idx} ({split})")

    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"prediction_{model_name}_idx{idx}_Inlen{input_length}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"✅ Saved plot to: {save_path}")
    
    
# RMSE Analysis
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from darts.metrics import rmse
from IPython.display import display

# def evaluate_and_save_results(
#     model,
#     timeseries_ds,
#     split_to_dataset,
#     split_name,
#     save_dir,
#     model_name=None,
#     model_type = 'NN' # 'linear'
# ):
#     """
#     Evaluate a Darts model on a split and save RMSE results and summaries.

#     Parameters:
#     - model: Fitted Darts model
#     - timeseries_ds: Dictionary with full/scaled series and scalers
#     - split_to_dataset: HuggingFace dataset with stratum column
#     - split_name: One of 'testid' or 'testod'
#     - save_dir: Path to store result files
#     - model_name: Optional string used in filenames (defaults to model.model_name)
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     model_name = model_name or getattr(model, 'model_name', 'model')
#     if model_type == 'NN':
#         input_chunk_length = model.input_chunk_length
#         output_chunk_length = model.output_chunk_length
#     else:
#         input_chunk_length = 288
#         output_chunk_length = 96
    
#     # Step 1: Predict efficiently in batch
#     full_series_list = timeseries_ds[split_name]['full_series_list']
#     full_series_scaled_list = timeseries_ds[split_name]['full_series_scaled_list']
#     scaler_list = timeseries_ds[split_name]['scaler_list']
#     input_series_scaled_list = [ts[:input_chunk_length] for ts in full_series_scaled_list]

#     print(f"⚙️ Predicting on {len(input_series_scaled_list)} series...")
#     pred_scaled_list = model.predict(n=output_chunk_length, series=input_series_scaled_list, verbose=False)

#     # Step 2: Compute RMSE metrics
#     results = []
#     for idx, (pred_scaled, scaler, full_series) in enumerate(zip(pred_scaled_list, scaler_list, full_series_list)):
#         pred = scaler.inverse_transform(pred_scaled)
#         true = full_series[-output_chunk_length:]
#         result_row = {'idx': idx}
#         for i in [6, 12, 24, 96]:
#             result_row[f'rmse_{i}'] = rmse(true[:i], pred[:i])
#         results.append(result_row)

#     results_df = pd.DataFrame(results)

#     # Step 3: Add stratum info (optional)
#     hf_key = split_name
#     if hf_key in split_to_dataset:
#         results_df['stratum'] = split_to_dataset[hf_key]['stratum']

#     # Step 4: Grouped summary statistics
#     if 'stratum' in results_df.columns:
#         summary_by_stratum = results_df.groupby("stratum")[["rmse_6", "rmse_12", "rmse_24", "rmse_96"]].describe()
#         summary_by_stratum.columns = ['_'.join(col).strip() for col in summary_by_stratum.columns.values]
#         summary_by_stratum = summary_by_stratum.round(2)

#         # Step 5: Save styled summary with color
#         cmap = mcolors.LinearSegmentedColormap.from_list("gwr", ["green", "white", "red"])
#         styled_summary = summary_by_stratum.style.background_gradient(cmap=cmap, axis=0)

#         styled_html_path = os.path.join(save_dir, f"{split_name}_{model_name}_summary.html")
#         styled_summary.to_html(styled_html_path)
#         print(f"📊 Saved styled summary to {styled_html_path}")

#         # Also save raw summary as CSV
#         summary_csv_path = os.path.join(save_dir, f"{split_name}_{model_name}_summary.csv")
#         summary_by_stratum.to_csv(summary_csv_path)
#     else:
#         styled_html_path = summary_csv_path = None
#         print("⚠️ Stratum not found in HuggingFace dataset; skipping grouped summary.")

#     # Step 6: Save per-sample results
#     raw_csv_path = os.path.join(save_dir, f"{split_name}_{model_name}_results.csv")
#     results_df.to_csv(raw_csv_path, index=False)
#     print(f"✅ Saved raw RMSE results to {raw_csv_path}")

#     # Optional: display in notebook
#     display(results_df.describe())
#     if styled_html_path:
#         display(styled_summary)

#     return results_df



