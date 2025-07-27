import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import datasets
from pprint import pprint
from datasets import disable_caching, load_dataset
from tqdm import tqdm
import optuna
from neuralforecast import NeuralForecast
from neuralforecast.models import (
    Informer, VanillaTransformer, Autoformer,
    LSTM, GRU, RNN
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, 
    format='[%(levelname)s:%(asctime)s:(%(filename)s@%(lineno)d %(name)s)]: %(message)s'
)

# Configure Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Define paths
SPACE = {
    'DATA_RAW': '_Data/0-Data_Raw',
    'DATA_RFT': '_Data/1-Data_RFT',
    'DATA_CASE': '_Data/2-Data_CASE',
    'DATA_AIDATA': '_Data/3-Data_AIDATA',
    'DATA_EXTERNAL': 'code/external',
    'DATA_HFDATA': '_Data/5-Data_HFData',
    'CODE_FN': 'code/pipeline',
    'MODEL_ROOT': '_Model',
}

assert os.path.exists(SPACE['CODE_FN']), f'{SPACE["CODE_FN"]} not found'
sys.path.append(SPACE['CODE_FN'])
from nn.eval.seqeval import SeqPredEval

# Configure logging
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
disable_caching()


COLUMNS_TO_USE = ['unique_id', 'ds', 'y']

def suggest_steps_from_epochs(trial, min_epochs, max_epochs, batch_size, total_samples):
    """🆕 NEW: Convert epochs to steps based on dataset size and batch size"""
    steps_per_epoch = total_samples / batch_size
    min_steps = int(min_epochs * steps_per_epoch)
    max_steps = int(max_epochs * steps_per_epoch)
    return trial.suggest_int('max_steps', min_steps, max_steps)


def get_hyperparameter_space(model_type, trial, total_samples):

    if model_type == 'informer':
        # 🔄 UPDATED: Suggest batch_size first
        batch_size = trial.suggest_categorical('batch_size', [64, 128])
        
        # 🆕 NEW: Calculate max_steps from desired epochs (8-20 epochs for complex transformer)
        max_steps = suggest_steps_from_epochs(trial, 8, 20, batch_size, total_samples)
        
        return {
            'h': trial.suggest_int('h', 1, 1),  # Will be set to output_length
            'input_size': trial.suggest_int('input_size', 1, 1),  # Will be set to input_length
            'hidden_size': trial.suggest_int('hidden_size', 64, 128, step=64),
            'n_head': trial.suggest_categorical('n_head', [4, 8]),
            'encoder_layers': trial.suggest_int('encoder_layers', 1, 2),
            'decoder_layers': trial.suggest_int('decoder_layers', 1, 2),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'max_steps': max_steps,  # 🔄 UPDATED: Now calculated from epochs
            'batch_size': batch_size,  # 🔄 UPDATED: Now used in max_steps calculation
        }

    elif model_type == 'autoformer':
        batch_size = trial.suggest_categorical('batch_size', [32, 64])
        
        max_steps = suggest_steps_from_epochs(trial, 8, 20, batch_size, total_samples)
        
        return {
            'h': trial.suggest_int('h', 1, 1),  # Will be set to output_length
            'input_size': trial.suggest_int('input_size', 1, 1),  # Will be set to input_length
            'hidden_size': trial.suggest_int('hidden_size', 32, 64, step=32),
            'n_head': trial.suggest_categorical('n_head', [4, 8]),
            'encoder_layers': trial.suggest_int('encoder_layers', 1, 2),
            'decoder_layers': trial.suggest_int('decoder_layers', 1, 2),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'max_steps': max_steps,  
            'batch_size': batch_size,  
        }


    elif model_type == 'transformer':
        batch_size = trial.suggest_categorical('batch_size', [32, 64])
        
        max_steps = suggest_steps_from_epochs(trial, 8, 20, batch_size, total_samples)
        
        return {
            'h': trial.suggest_int('h', 1, 1),  # Will be set to output_length
            'input_size': trial.suggest_int('input_size', 1, 1),  # Will be set to input_length
            'hidden_size': trial.suggest_int('hidden_size', 32, 64, step=32),
            'n_head': trial.suggest_categorical('n_head', [4, 8]),
            'encoder_layers': trial.suggest_int('encoder_layers', 1, 3),
            'decoder_layers': trial.suggest_int('decoder_layers', 1, 3),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'max_steps': max_steps,  
            'batch_size': batch_size,  
        }


    elif model_type in ['lstm', 'gru', 'rnn']:
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
        
        max_steps = suggest_steps_from_epochs(trial, 5, 15, batch_size, total_samples)
        
        return {
            'h': trial.suggest_int('h', 1, 1),  # Will be set to output_length
            'input_size': trial.suggest_int('input_size', 1, 1),  # Will be set to input_length
            
            
            'encoder_hidden_size': trial.suggest_int('encoder_hidden_size', 64, 512, step=64),
            'decoder_hidden_size': trial.suggest_int('decoder_hidden_size', 64, 512, step=64),
            'encoder_n_layers': trial.suggest_int('encoder_n_layers', 1, 4),
            'decoder_layers': trial.suggest_int('decoder_layers', 1, 4),
            'encoder_dropout': trial.suggest_float('encoder_dropout', 0.0, 0.3),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'max_steps': max_steps,  # 🔄 UPDATED: Now calculated from epochs
            'batch_size': batch_size,  # 🔄 UPDATED: Now used in max_steps calculation
        }

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def create_model_with_params(model_type, params):
    """Create neural network model with specific parameters"""
    
    # Neural Network Models
    neural_models = {
        'informer': Informer,
        'transformer': VanillaTransformer,
        'autoformer': Autoformer,
        'lstm': LSTM,
        'gru': GRU,
        'rnn': RNN,
    }

    if model_type not in neural_models:
        raise ValueError(f"Unsupported neural model type: {model_type}")
    
    # Filter out trainer-incompatible parameters
    disallowed_trainer_args = {'lstm_layers', 'stack_type_1', 'stack_types'}
    model_class = neural_models[model_type]

    # Remove invalid block types for nhits
    if model_type == 'nhits':
        invalid_stack_types = {'polynomial_trend'}
        if 'stack_type_1' in params and params['stack_type_1'] in invalid_stack_types:
            print(f"⚠️ Warning: Invalid block type '{params['stack_type_1']}' for NHITS — removing.")
            del params['stack_type_1']

    # Separate trainer kwargs if needed in future (currently unused)
    model_kwargs = {k: v for k, v in params.items() if k not in disallowed_trainer_args}

    model = model_class(**model_kwargs)
    return NeuralForecast(models=[model], freq='5min')




def generate_prediction_nixtla_one_dfeval(model, df_eval, CURRENT_POINT, INPUT_LENGTH, OUTPUT_LENGTH):
    # --------------------------------
    # COLUMNS_TO_USE = ['unique_id', 'ds', 'y']
    start = CURRENT_POINT - INPUT_LENGTH
    end = CURRENT_POINT + OUTPUT_LENGTH

    # Slice the input and real target windows
    df_eval_input = df_eval[(df_eval['abs_step'] >= start) & (df_eval['abs_step'] < CURRENT_POINT)]
    df_eval_real  = df_eval[(df_eval['abs_step'] >= CURRENT_POINT) & (df_eval['abs_step'] < end)]

    # Run model prediction
    df_eval_pred = model.predict(df_eval_input[COLUMNS_TO_USE])

    # Show shapes
    # print(df_eval_input.shape, df_eval_real.shape, df_eval_pred.shape)

    # === FIXED: avoid FutureWarning by selecting columns explicitly ===

    # Extract past input series
    df_input = (
        df_eval_input.groupby('unique_id')['y']
        .apply(list)
        .reset_index(name='Inputs')
    )

    # Extract ground-truth future series
    df_real = (
        df_eval_real.groupby('unique_id')['y']
        .apply(list)
        .reset_index(name='Real')
    )

    # Detect prediction column
    model_names = [col for col in df_eval_pred.columns if col not in COLUMNS_TO_USE]
    if not model_names:
        raise ValueError("No model prediction column found")
    model_name = model_names[0]

    # Extract predicted series
    df_pred = (
        df_eval_pred.groupby('unique_id')[model_name]
        .apply(list)
        .reset_index(name='Predict')
    )

    # Extract the time of prediction (last input timestamp)
    df_prediction_time = (
        df_eval_input.groupby('unique_id')['ds']
        .max()
        .reset_index(name='prediction_time')
    )

    # Merge into one dataframe
    df_predict = pd.merge(df_input, df_real, on='unique_id', how='left')
    df_predict = pd.merge(df_predict, df_pred, on='unique_id', how='left')
    df_predict = pd.merge(df_predict, df_prediction_time, on='unique_id', how='left')

    # (Optional) Add back stratum if needed
    df_stratum = df_eval[['unique_id', 'stratum']].drop_duplicates().reset_index(drop=True)
    df_predict = pd.merge(df_predict, df_stratum, on='unique_id', how='left')

    # Display result
    return df_predict


def optimize_hyperparameters(model_type, CURRENT_POINT, INPUT_LENGTH, OUTPUT_LENGTH,
                            train_df, val_df, n_trials=20):
    """🔄 UPDATED: Optimize hyperparameters using Optuna"""
    
    # 🆕 NEW: Calculate total samples for epoch-based max_steps calculation
    total_samples = len(train_df['unique_id'].unique())
    logger.info(f"🆕 Total training samples: {total_samples:,}")
    
    def objective(trial):
        
        params = get_hyperparameter_space(model_type, trial, total_samples)
        logger.info(f"🆕 Trial {trial.number}: params = {params}")
        # Set fixed parameters
        params['h'] = OUTPUT_LENGTH
        params['input_size'] = INPUT_LENGTH
        
        # 🆕 NEW: Log epoch information for transparency
        steps_per_epoch = total_samples / params['batch_size']
        effective_epochs = params['max_steps'] / steps_per_epoch
        logger.info(f"🆕 Trial {trial.number}: {params['max_steps']} steps = "
                    f"{effective_epochs:.1f} epochs (batch_size={params['batch_size']})")
        
        # Create and train model

        model = create_model_with_params(model_type, params)
        
        # Train on training data
        try:
            model.fit(train_df[COLUMNS_TO_USE])
        except Exception as e:
            logger.error(f"🆕 Trial {trial.number}: Error fitting model: {e}")
            return float('inf')
        
        # Predict on validation data
        # df_predict = generate_predictions(model, val_df, CURRENT_POINT, INPUT_LENGTH, OUTPUT_LENGTH)
        df_predict = generate_prediction_nixtla_one_dfeval(model, val_df, CURRENT_POINT, INPUT_LENGTH, OUTPUT_LENGTH)

        def compute_rmse(real, pred):
            real = np.array(real)
            pred = np.array(pred)
            return np.sqrt(np.mean((real - pred) ** 2))

        s = df_predict.apply(lambda x: compute_rmse(x['Real'], x['Predict']), axis=1)
        rMSE = s.mean()
        logger.info(f"🆕 Trial {trial.number}: params = {params}")
        logger.info(f"🆕 Trial {trial.number}: rMSE = {rMSE}")
        return rMSE
        
    # Create study and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials = n_trials)
    
    # 🆕 NEW: Enhanced logging with epoch information
    best_params = study.best_params
    if 'batch_size' in best_params and 'max_steps' in best_params:
        best_steps_per_epoch = total_samples / best_params['batch_size']
        best_effective_epochs = best_params['max_steps'] / best_steps_per_epoch
        
        logger.info(f"🆕 Best configuration: {best_params['max_steps']} steps = "
                   f"{best_effective_epochs:.1f} epochs (batch_size={best_params['batch_size']})")
    
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value}")
    logger.info(f"Best params: {study.best_params}")
    
    return study.best_params


def create_model(model_type, CURRENT_POINT, INPUT_LENGTH, OUTPUT_LENGTH, train_df, val_df, n_trials=20):
    logger.info(f"Starting hyperparameter optimization for {model_type} with {n_trials} trials...")
    
    # Optimize hyperparameters
    best_params = optimize_hyperparameters(model_type, CURRENT_POINT, INPUT_LENGTH, OUTPUT_LENGTH,
                                         train_df, val_df, n_trials)
    
    # Set fixed parameters
    best_params['h'] = OUTPUT_LENGTH
    best_params['input_size'] = INPUT_LENGTH
    
    logger.info(f"Creating final model with optimized parameters:")
    pprint(best_params)
    
    # Create final model with best parameters
    model = create_model_with_params(model_type, best_params)
    
    return model


def train_or_load_model(model_type, 
                        Name_to_nixtlaDF, 
                        to_train, 
                        model_path,
                        CURRENT_POINT,
                        INPUT_LENGTH,
                        OUTPUT_LENGTH,
                        n_trials):
    
    """Train or load neural network model"""
    save_model_path = os.path.join(model_path, f"model.pkl")

    df_train = Name_to_nixtlaDF['train']
    df_valid = Name_to_nixtlaDF['valid']
    
    input_length  = INPUT_LENGTH
    output_length = OUTPUT_LENGTH 
    start = CURRENT_POINT - input_length
    end   = CURRENT_POINT + output_length 

    df_train_filtered = df_train[(df_train['abs_step'] >= start) & (df_train['abs_step'] < end)].reset_index(drop=True)
    df_valid_filtered = df_valid[(df_valid['abs_step'] >= start) & (df_valid['abs_step'] < end)].reset_index(drop=True)

    if to_train:
        logger.info(f"Training neural network model with hyperparameter optimization...")
        
        # Prepare training and validation data for neural model
        train_df = df_train_filtered.copy()# [['unique_id', 'ds', 'y']].copy()
        val_df = df_valid_filtered.copy()# [['unique_id', 'ds', 'y']].copy()
        
        # Create model with hyperparameter optimization
        model = create_model(model_type, CURRENT_POINT, INPUT_LENGTH, OUTPUT_LENGTH, train_df, val_df, n_trials)
        
        # Train final model on all training data
        model.fit(train_df[COLUMNS_TO_USE])
        
        # Save model
        if hasattr(model, 'save'):
            model.save(save_model_path, overwrite=True)
        else:
            import pickle
            with open(save_model_path, 'wb') as f:
                pickle.dump(model, f)
        logger.info(f"Model saved to: {save_model_path}")
    else:
        # Load model
        if not os.path.exists(save_model_path):
            raise FileNotFoundError(f"Model not found at: {save_model_path}")
        
        if hasattr(model, 'load'):
            model = model.load(save_model_path)
        else:
            import pickle
            with open(save_model_path, 'rb') as f:
                model = pickle.load(f)
        logger.info(f"Model loaded from: {save_model_path}")
    
    return model


def generate_predictions(model, 
                         Name_to_nixtlaDF, 
                         CURRENT_POINT,
                         INPUT_LENGTH,
                         OUTPUT_LENGTH,
                         evalset_names,
                         ):
    """Generate predictions for all splits"""

    
    all_data = []
    
    for split_name in evalset_names:
        logger.info(f"Processing split: {split_name}")
        
        df_eval = Name_to_nixtlaDF[split_name]
        # df_predict = generate_predictions(model, df_eval, CURRENT_POINT, INPUT_LENGTH, OUTPUT_LENGTH)
        df_predict = generate_prediction_nixtla_one_dfeval(model, df_eval, CURRENT_POINT, INPUT_LENGTH, OUTPUT_LENGTH)

        df_predict['Splitname'] = split_name
        all_data.append(df_predict)

    df_predict = pd.concat(all_data).reset_index(drop=True)
    return df_predict


def evaluate_predictions(df_predict, subgroup_config_list, horizon_to_se, metric_list):
    """Evaluate predictions using SeqPredEval"""
    # Load and preprocess data
    try:
        df_predict['prediction_hour'] = (df_predict['prediction_time'] / 12).astype(int)
    except:
        df_predict['prediction_hour'] = df_predict['prediction_time'].dt.hour 

    try:
        df_predict['age_interval'] = df_predict['stratum'].apply(lambda x: x.split('_')[0])
        df_predict['gender'] = df_predict['stratum'].apply(lambda x: x.split('_')[1])
        df_predict['diabetes_type'] = df_predict['stratum'].apply(lambda x: x.split('_')[2])
    except:
        df_predict['age_interval'] = 'unknown'
        df_predict['gender'] = 'unknown'
        df_predict['diabetes_type'] = 'T1DM'

    # Configure evaluation parameters
    x_hist_seq_name = 'Inputs'
    y_real_seq_name = 'Real'
    y_pred_seq_name = 'Predict'
    
    # Filter and prepare data for evaluation
    df_predict_to_eval = df_predict
    df_case_eval = df_predict_to_eval

    # Create evaluation instance and run evaluation
    eval_instance = SeqPredEval(
        df_case_eval=df_case_eval,
        subgroup_config_list=subgroup_config_list,
        x_hist_seq_name=x_hist_seq_name,
        y_real_seq_name=y_real_seq_name,
        y_pred_seq_name=y_pred_seq_name,
        metric_list=metric_list,
        horizon_to_se=horizon_to_se,
    )

    # Get evaluation results
    df_report_neat = eval_instance.df_report_neat.copy()
    df_report_full = eval_instance.df_report_full.copy()

    return df_report_neat, df_report_full


CURRENT_POINT = 289 

# Evaluation configuration
subgroup_config_list = [
    ['Splitname'],
    # ['Splitname', 'age_interval'],
    # ['Splitname', 'gender'],
    # ['Splitname', 'diabetes_type'],
    # ['Splitname', 'stratum'],
    # ['Splitname', 'prediction_hour'],
]

horizon_to_se = {
    '000-030min': [0, 6],
    '000-060min': [0, 12],
    '000-120min': [0, 24],
    # '000-480min': [0, 96],
    # '060-120min': [12, 24],
}

metric_list = ['rMSE'] 
# Additional metrics: 'TIR_error', 'RegionDistError'


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='Run Neural Network Time Series Forecasting with Optuna hyperparameter optimization.')
    parser.add_argument('--input_length', type=int, default=36,
                        help='Input chunk length for the model. Common values: 288, 144, 72, 36.')
    parser.add_argument('--output_length', type=int, default=12,
                        help='Output chunk length for the model.')
    parser.add_argument('--model_type', type=str, default='patchtst',
                        choices=[
                            # Neural models
                            'informer', 'transformer', 'autoformer', 
                            'lstm', 'gru', 'rnn',
                        ],
                        help='Type of neural network model to use for training.')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of Optuna trials for hyperparameter optimization.')
    parser.add_argument('--no_train', action='store_false', dest='to_train',
                        help='If set, model will be loaded instead of trained.')

    args = parser.parse_args()

    INPUT_LENGTH = args.input_length
    OUTPUT_LENGTH = args.output_length
    MODEL_TYPE = args.model_type
    TO_TRAIN = args.to_train
    N_TRIALS = args.n_trials

    ######################## Load Data ########################
    NixDataName = 'OhioT1DM-Nixtla-v0628'
    data_path = os.path.join(SPACE['DATA_HFDATA'], NixDataName)
    
    # Add basic validation
    if not os.path.exists(data_path):
        logger.error(f"Data directory not found: {data_path}")
        sys.exit(1)

    Name_to_nixtlaDF = {}
    Name_list = [i.split('.parquet')[0] for i in os.listdir(data_path)]
    for Name in Name_list:
        file_path = os.path.join(SPACE['DATA_HFDATA'], NixDataName, f'{Name}.parquet')
        print(file_path)
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            Name_to_nixtlaDF[Name] = df
        else:
            logger.warning(f"File not found: {file_path}")

    for Name, df in Name_to_nixtlaDF.items():
        print(Name, df.shape)
    #############################################################

    """Main execution function"""
    logger.info(f"Processing input length: {INPUT_LENGTH}")

    # Setup model folder
    model_folder = os.path.join(SPACE['MODEL_ROOT'], 'glucopred_nix_optuna')
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, 
                               f"neural_{MODEL_TYPE}_i{INPUT_LENGTH}o{OUTPUT_LENGTH}_trials{N_TRIALS}")
    os.makedirs(model_path, exist_ok=True)

    # Train or load model
    model = train_or_load_model(MODEL_TYPE,
                                Name_to_nixtlaDF, 
                                TO_TRAIN, 
                                model_path,
                                CURRENT_POINT,
                                INPUT_LENGTH, 
                                OUTPUT_LENGTH,
                                N_TRIALS)
    
    print(f"Neural Model with Optuna optimization: {model}")
    
    # Generate sample plots
    plot_dir = os.path.join(model_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Generate and save predictions
    evalset_names = ['test-id',]
    df_predict = generate_predictions(model, 
                                      Name_to_nixtlaDF, 
                                      CURRENT_POINT,
                                      INPUT_LENGTH, 
                                      OUTPUT_LENGTH,
                                      evalset_names)
    
    # Print summary statistics
    print(f"df_predict shape: {df_predict.shape}")

    # Save results
    # results_dir = os.path.join(os.path.dirname(model_folder), "results")
    # os.makedirs(results_dir, exist_ok=True)
    
    output_file = os.path.join(model_path, "df_predictions.parquet")
    df_predict.to_parquet(output_file, index=False)
    logger.info(f"Results saved to: {output_file}")

    # Evaluate predictions
    filtered_horizon_to_se = {k: v for k, v in horizon_to_se.items() if v[1] <= OUTPUT_LENGTH}
    df_report_neat, df_report_full = evaluate_predictions(df_predict, subgroup_config_list, filtered_horizon_to_se, metric_list)
        
    print("Evaluation Report (Neat):")
    print(df_report_neat)
    print("\nEvaluation Report (Full):")
    print(df_report_full)

    # Save evaluation results
    eval_output_file = os.path.join(model_path, f"eval.parquet")
    df_report_full.to_parquet(eval_output_file)
    logger.info(f"Evaluation results saved to: {eval_output_file}")


"""
python 2-OhioT1DM/run_glucopred_models.py --model_type lstm --input_length 289 --output_length 24 --n_trials 10
python 2-OhioT1DM/run_glucopred_models.py --model_type gru --input_length 289 --output_length 24 --n_trials 10
python 2-OhioT1DM/run_glucopred_models.py --model_type rnn --input_length 289 --output_length 24 --n_trials 10
python 2-OhioT1DM/run_glucopred_models.py --model_type informer --input_length 289 --output_length 24 --n_trials 10
python 2-OhioT1DM/run_glucopred_models.py --model_type autoformer --input_length 289 --output_length 24 --n_trials 10
python 2-OhioT1DM/run_glucopred_models.py --model_type transformer --input_length 289 --output_length 24 --n_trials 10
"""