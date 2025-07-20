import os
import sys
import pickle
import inspect
import importlib 
import yaml
import numpy as np
import pandas as pd
import logging
import datasets
from functools import reduce
import pprint 
import os
import shutil
import tarfile

logger = logging.getLogger(__name__)


# Function to copy all folders from code_path to path_endpoint
def copy_folders(source_path, destination_path):
    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
        print(f"Created destination directory: {destination_path}")
    
    # Get list of all folders in source path
    folders_to_copy = [f for f in os.listdir(source_path) 
                      if os.path.isdir(os.path.join(source_path, f))]
    
    copied_folders = []
    
    # Copy each folder
    for folder in folders_to_copy:
        source_folder = os.path.join(source_path, folder)
        dest_folder = os.path.join(destination_path, folder)
        
        try:
            # If destination folder already exists, remove it
            if os.path.exists(dest_folder):
                shutil.rmtree(dest_folder)
            
            # Copy the folder to destination
            shutil.copytree(source_folder, dest_folder)
            
            # Add to list of successfully copied folders
            copied_folders.append(folder)
            print(f"Copied folder: {folder}")
            
        except Exception as e:
            print(f"Error copying folder {folder}: {str(e)}")
    
    print(f"Successfully copied {len(copied_folders)} folders to {destination_path}")
    return copied_folders


def cleanup_folder(folder_path, custom_patterns=None):
    """
    Remove unwanted files and directories before uploading to S3.
    
    :param folder_path: Path to the folder to clean
    :param custom_patterns: Additional patterns to remove (optional)
    :return: tuple (number of files removed, number of directories removed)
    """
    import glob
    import shutil
    from pathlib import Path
    
    # Default patterns to remove
    default_patterns = {
        'files': [
            '*.pyc',           # Python compiled files
            '.DS_Store',       # Mac OS system files
            '*.log',           # Log files
            '*.tmp',           # Temporary files
            '*~',              # Backup files
            '*.swp',           # Vim swap files
            '.env',            # Environment files
            '*.bak',           # Backup files
            'Thumbs.db',       # Windows thumbnail cache
            '*.class',         # Java compiled files
        ],
        'dirs': [
            '__pycache__',     # Python cache directories
            '.ipynb_checkpoints', # Jupyter notebook checkpoints
            '.pytest_cache',    # Pytest cache
            '.git',            # Git directory
            '.idea',           # PyCharm files
            '.vscode',         # VSCode files
            'node_modules',    # Node.js modules
            'venv',            # Python virtual environment
            'env',             # Another common venv name
        ]
    }
    
    # Add custom patterns if provided
    if custom_patterns:
        if isinstance(custom_patterns, (list, tuple)):
            default_patterns['files'].extend(custom_patterns)
        elif isinstance(custom_patterns, dict):
            default_patterns['files'].extend(custom_patterns.get('files', []))
            default_patterns['dirs'].extend(custom_patterns.get('dirs', []))
    
    files_removed = 0
    dirs_removed = 0
    
    try:
        # Convert to Path object for better path handling
        folder_path = Path(folder_path)
        
        # First remove unwanted directories
        for dir_pattern in default_patterns['dirs']:
            for item in folder_path.rglob(dir_pattern):
                if item.is_dir():
                    shutil.rmtree(item)
                    print(f"Removed directory: {item}")
                    dirs_removed += 1
        
        # Then remove unwanted files
        for file_pattern in default_patterns['files']:
            for item in folder_path.rglob(file_pattern):
                if item.is_file():
                    item.unlink()
                    print(f"Removed file: {item}")
                    files_removed += 1
        
        print(f"\nCleanup completed successfully!")
        print(f"Removed {files_removed} files and {dirs_removed} directories")
        
        return files_removed, dirs_removed
        
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        raise


def copy_code_to_model_path(code_path, model_endpoint_path):
    """
    Copy code from the source path to the model endpoint path.
    
    Args:
        project_folder (str): The project folder path
        model_endpoint_path (str): The destination model endpoint path
    
    Returns:
        None
    """
    # Define the source code path
    # code_path = os.path.join(project_folder, '000-Sagemaker-Pipeline/_Data_from_s3/code')
    
    # Create the destination directory if it doesn't exist
    # os.makedirs(model_endpoint_path, exist_ok=True)
    assert os.path.exists(code_path), f"Source code path {code_path} does not exist"
    assert os.path.exists(model_endpoint_path), f"Destination model endpoint path {model_endpoint_path} does not exist"
    
    # Copy the code folder to the model endpoint path
    if os.path.exists(code_path):
        # Copy all files from code_path to model_endpoint_path
        for item in os.listdir(code_path):
            source_item = os.path.join(code_path, item)
            dest_item = os.path.join(model_endpoint_path, item)
            
            if os.path.isdir(source_item):
                # If it's a directory, copy the entire directory
                if os.path.exists(dest_item):
                    shutil.rmtree(dest_item)
                shutil.copytree(source_item, dest_item)
            else:
                # If it's a file, copy the file
                shutil.copy2(source_item, dest_item)
        
        print(f"Successfully copied code from {code_path} to {model_endpoint_path}")
    else:
        print(f"Warning: Source code path {code_path} does not exist")


def clean_and_archive_model(model_endpoint_path):
    """
    Cleans up unwanted files and directories from the model artifact path
    and creates a tar.gz archive of the remaining contents.
    
    Args:
        model_root (str): The root directory where models are stored.
        model_version (str): The version of the model to be archived.
    """
    # model_artifact_path = os.path.join(model_root, model_version)
    print(f'model_endpoint_path: {model_endpoint_path}')
    cleanup_folder(model_endpoint_path)
    
    
    # List of unwanted folder names and file patterns
    unwanted_folders = ['__pycache__']
    unwanted_files = ['.DS_Store', '.pyc', '.pyo', 'thumbs.db']
    mac_prefix = '._'  # Prefix used by macOS for resource fork files

    
    # Traverse the model artifact path
    for root, dirs, files in os.walk(model_endpoint_path):
        # Remove unwanted directories
        for dir_name in dirs:
            if dir_name.startswith('_.') or dir_name in unwanted_folders:
                dir_path = os.path.join(root, dir_name)
                shutil.rmtree(dir_path)
                print(f'Deleted folder: {dir_path}')
        
        # Remove unwanted files
        for file_name in files:
            if (file_name in unwanted_files or file_name.startswith(mac_prefix)):
                file_path = os.path.join(root, file_name)
                os.remove(file_path)
                print(f'Deleted file: {file_path}')
    
    # Create a tar.gz archive of the remaining contents
    tar_path = f'{model_endpoint_path}.tar.gz'
    with tarfile.open(tar_path, 'w:gz') as tar:
        tar.add(model_endpoint_path, arcname=os.path.basename(model_endpoint_path))
    
    print(f'Created tar.gz archive: {tar_path}')
    return tar_path



def download_s3_folder_or_file(S3_CLIENT, BUCKET_NAME, S3_BASE_PATH, local_folder_or_file_path):
    """
    Download a folder from S3 to a local directory.
    :param s3_client: Boto3 S3 client
    :param bucket_name: Name of the S3 bucket
    :param s3_folder: S3 folder path (prefix)
    :param local_dir: Local directory to download the folder to
    """

    if '.' not in os.path.basename(local_folder_or_file_path):
        local_folder_path = local_folder_or_file_path
        s3_folder = os.path.join(S3_BASE_PATH, local_folder_path)
        file_name = None
        # s3_file = None 
        print(f'downloading the folder: {s3_folder} to: {local_folder_path}')
    else:
        local_folder_path = os.path.dirname(local_folder_or_file_path)
        file_name = os.path.basename(local_folder_or_file_path)
        s3_folder = os.path.join(S3_BASE_PATH, local_folder_path)
        # s3_file = os.path.join(s3_folder, file_name)
        print(f'downloading the file: {s3_folder} to: {local_folder_path}')

    

    # Ensure the local directory exists
    if not os.path.exists(local_folder_path):
        os.makedirs(local_folder_path)
        print(f'create the local_folder_path: {local_folder_path}')

    # List objects in the specified S3 folder
    response = S3_CLIENT.list_objects_v2(Bucket=BUCKET_NAME, Prefix=s3_folder)

    # Download each object

    downloaded_count = 0
    if 'Contents' in response:

        for obj in response['Contents']:
            # Get the object key
            s3_key = obj['Key']
            # print(f'downloading the file: {s3_key}')
            if s3_key.endswith('/'):  # Add any other unwanted patterns as needed
                continue  # Skip invalid keys
        
            if file_name is not None:
                if os.path.basename(s3_key) != file_name: continue 
            
            # Create a local file path
            local_file_path = os.path.join(local_folder_path, os.path.relpath(s3_key, s3_folder))

            if '/../' in local_file_path:
                continue 

            print('copying the file: ', s3_key)
            print('      ------- to: ', local_file_path)
            # Ensure the local directory structure exists
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            # Download the file
            S3_CLIENT.download_file(BUCKET_NAME, s3_key, local_file_path)
            downloaded_count += 1

    print(f'Download complete: {downloaded_count} files downloaded out of {len(response["Contents"])} total files')
    return downloaded_count


def upload_s3_folder_or_file(S3_CLIENT, BUCKET_NAME, S3_BASE_PATH, local_folder_or_file_path):
    """
    Upload a local directory to S3.
    
    :param S3_CLIENT: Boto3 S3 client
    :param BUCKET_NAME: Name of the S3 bucket
    :param S3_BASE_PATH: S3 folder path (prefix)
    :param local_folder_path: Local directory to upload
    """
    # Ensure the local directory exists


    if not os.path.exists(local_folder_or_file_path):
        raise ValueError(f"Local directory '{local_folder_or_file_path}' does not exist")
    

    if os.path.isdir(local_folder_or_file_path):
        local_folder_path = local_folder_or_file_path
        s3_folder = os.path.join(S3_BASE_PATH, local_folder_path)
        file_name = None
    else:
        local_folder_path = os.path.dirname(local_folder_or_file_path)
        file_name = os.path.basename(local_folder_or_file_path)

    # Walk through the local directory
    file_counts = 0
    for root, dirs, files in os.walk(local_folder_path):
        for filename in files:
            # Get the full local path

            if file_name is not None:
                if os.path.basename(filename) != file_name: continue 

            local_path = os.path.join(root, filename)
            
            # Calculate relative path from the local_dir
            # relative_path = os.path.relpath(local_path, local_folder_path)
            
            # Create the S3 key by joining the s3_folder with the relative path
            s3_key = os.path.join(S3_BASE_PATH, local_path).replace("\\", "/")
            
            # Upload the file
            print(f'Uploading {local_path} to s3://{BUCKET_NAME}/{s3_key}')
            S3_CLIENT.upload_file(local_path, BUCKET_NAME, s3_key)
            file_counts += 1

    print(f'Upload complete: {file_counts} files uploaded')
    return file_counts





def delete_s3_folder_or_file(S3_CLIENT, 
                            BUCKET_NAME, 
                            S3_BASE_PATH, 
                            local_folder_or_file_path):
    """
    Delete all objects in an S3 folder/prefix.
    
    :param S3_CLIENT: Boto3 S3 client
    :param BUCKET_NAME: Name of the S3 bucket
    :param S3_BASE_PATH: S3 base path (prefix)
    :param local_folder_or_file_path: Local directory or file path to determine what to delete in S3
    """

    if os.path.isdir(local_folder_or_file_path):
        local_folder_path = local_folder_or_file_path
        # relative_path = os.path.basename(local_folder_path)
        s3_folder = os.path.join(S3_BASE_PATH, local_folder_path)
    else:
        local_folder_path = os.path.dirname(local_folder_or_file_path)
        file_name = os.path.basename(local_folder_or_file_path)
        # relative_folder = os.path.relpath(local_folder_path, os.path.dirname(local_folder_path))
        s3_folder = os.path.join(S3_BASE_PATH, local_folder_path)
        s3_key = os.path.join(s3_folder, file_name).replace("\\", "/")
        
        # If it's a file, delete just that specific file
        try:
            print(f'Deleting file s3://{BUCKET_NAME}/{s3_key}')
            S3_CLIENT.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
            return
        except Exception as e:
            print(f"Error while deleting file: {str(e)}")
            raise

    # Ensure s3_folder ends with '/' if it's meant to be a folder
    if not s3_folder.endswith('/'):
        s3_folder += '/'
        
    # List all objects within the folder
    paginator = S3_CLIENT.get_paginator('list_objects_v2')
    objects_to_delete = []
    
    try:
        # Iterate through all objects in the folder
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_folder):
            if 'Contents' in page:
                # Collect all object keys
                objects_to_delete.extend([{'Key': obj['Key']} for obj in page['Contents']])
                
        if objects_to_delete:
            # Delete objects in batches of 1000 (S3 limit)
            for i in range(0, len(objects_to_delete), 1000):
                batch = objects_to_delete[i:i + 1000]
                response = S3_CLIENT.delete_objects(
                    Bucket=BUCKET_NAME,
                    Delete={
                        'Objects': batch,
                        'Quiet': True
                    }
                )
                print(f'Deleted {len(batch)} objects from s3://{BUCKET_NAME}/{s3_folder}')
                
                # Check for errors
                if 'Errors' in response:
                    for error in response['Errors']:
                        print(f"Error deleting {error['Key']}: {error['Message']}")
        else:
            print(f"No objects found in s3://{BUCKET_NAME}/{s3_folder}")
            print("Do not need to delete anything")
            
    except Exception as e:
        print(f"Error while deleting folder: {str(e)}")
        raise
    

def retrieve_name(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var and var_name[0] != '_']
        if len(names) > 0:
            return names[0]
        

def replace_none_with_list(d):
    for key, value in d.items():
        if value is None:
            d[key] = []
        elif isinstance(value, dict):
            replace_none_with_list(value)
    return d


def fill_missing_keys(input_form, template):
    # Create a copy of the template to preserve the original structure
    filled_form = template.copy()
    
    # Function to recursively fill missing keys based on the template
    def recurse_fill(current_form, current_template):
        if isinstance(current_template, dict):
            # Ensure current_form is a dictionary
            if not isinstance(current_form, dict):
                current_form = {}
            for key in current_template:
                if isinstance(current_template[key], dict):
                    # If the template value is a dictionary, recurse into it
                    current_form[key] = recurse_fill(current_form.get(key, {}), current_template[key])
                elif isinstance(current_template[key], list) and current_template[key] and isinstance(current_template[key][0], dict):
                    # If the template value is a list of dictionaries, process each dictionary
                    if key in current_form and isinstance(current_form[key], list):
                        # Ensure each element in the input list conforms to the template
                        current_form[key] = [recurse_fill(elem, current_template[key][0]) for elem in current_form[key]]
                    else:
                        # If the key is missing or not a list in the input, initialize it with a list containing a filled template dict
                        current_form[key] = [recurse_fill({}, current_template[key][0])]
                else:
                    # Set the key to None if it is not present in the input and not a dict or list of dicts
                    current_form[key] = current_form.get(key, None)
            return current_form
        elif isinstance(current_template, list) and current_template and isinstance(current_template[0], dict):
            # If the top-level template itself is a list of dictionaries
            if isinstance(current_form, list):
                return [recurse_fill(elem, current_template[0]) for elem in current_form]
            else:
                return [recurse_fill({}, current_template[0])]
        else:
            # Return None for unexpected types or if the template does not contain a dict or list of dicts
            return None

    # Call the recursive function starting with the entire form
    filled_form = recurse_fill(input_form, filled_form)
    return filled_form



###### ------------ part 2.2: filtering method ------------ ######
def apply_condition(df, column, operator, value):
    if column not in df.columns:
        logger.warning(f"Column <{column}> not in dataframe, pass this checking")
        return pd.Series([True] * len(df))
    
    if type(value) == str : 
        if 'lambda' in value:
            print(value)
            value = eval(value) 
            print(value)

    if callable(value):
        value = value(df)
    
    if operator == '>':
        return df[column] > value
    elif operator == '<':
        return df[column] < value
    elif operator == '==':
        return df[column] == value
    elif operator == '!=':
        return df[column] != value
    elif operator == '>=':
        return df[column] >= value
    elif operator == '<=':
        return df[column] <= value
    elif operator == 'contains':
        return df[column].astype(str).str.contains(value)
    elif operator == 'in':
        return df[column].isin(value)
    else:
        raise ValueError("Unsupported operator")
    

def apply_multiple_conditions(df, conditions, logical_operator='and'):
    if not conditions:
        raise ValueError("No conditions provided")

    # Initialize the result with the first condition
    condition = conditions[0]
    result = apply_condition(df, *condition)
    logger.info(f'condition: {condition}: to select: {result.sum()} out of {len(result)}')
    # Apply each subsequent condition
    for condition in conditions[1:]:
        new_condition = apply_condition(df, *condition)
        logger.info(f'condition: {condition}: to select: {new_condition.sum()} out of {len(new_condition)}')
        if logical_operator == 'and':
            result = result & new_condition
        elif logical_operator == 'or':
            result = result | new_condition
        else:
            raise ValueError("Unsupported logical operator")
    return result


###### ------------ part 2.3: spliting method ------------ ######
def generate_random_tags(df_case, RANDOM_SEED, HumanID, ObsDT):
    np.random.seed(RANDOM_SEED)
    df_case['RandDownSample'] = np.random.rand(len(df_case))

    # in&out
    df_P = df_case[[HumanID]].drop_duplicates().reset_index(drop = True)
    np.random.seed(RANDOM_SEED + 1)
    df_P['RandInOut'] = np.random.rand(len(df_P))
    df_case = pd.merge(df_case, df_P, how = 'left')

    # test
    df_case['CaseLocInP'] = df_case.groupby(HumanID).cumcount()
    df_case = pd.merge(df_case, df_case[HumanID].value_counts().reset_index())
    df_case['CaseRltLocInP'] = df_case['CaseLocInP'] /  df_case['count']
    
    # test other options
    np.random.seed(RANDOM_SEED + 2)
    df_case['RandTest'] = np.random.rand(len(df_case))

    # validation
    np.random.seed(RANDOM_SEED + 3)
    df_case['RandValidation'] = np.random.rand(len(df_case))

    df_case = df_case.drop(columns = ['CaseLocInP', 'count']).reset_index(drop = True)
    df_case = df_case.sort_values('RandDownSample').reset_index(drop = True)

    random_columns = [# 'RandDownSample', 
                      'RandInOut', 'CaseRltLocInP', 'RandTest', 'RandValidation']
    return df_case, random_columns


def assign_caseSplitTag_to_dsCase(df_case, 
                                  RANDOM_SEED, 
                                  HumanID, 
                                  ObsDT,
                                  out_ratio, 
                                  test_ratio, 
                                  valid_ratio, 
                                  **kwargs):

    df = df_case 
    df_rs, random_columns = generate_random_tags(df, RANDOM_SEED, HumanID, ObsDT,)
    df_dsmp = df_rs# [df_rs['RandDownSample'] <= downsample_ratio].reset_index(drop = True)

    df_dsmp['Out'] = df_dsmp['RandInOut'] < out_ratio
    df_dsmp['In'] = df_dsmp['RandInOut'] >= out_ratio
    assert df_dsmp[['Out', 'In']].sum(axis = 1).mean() == 1
    # logger.info(f"Out ratio: {df_dsmp['Out'].mean()}")
    # print(df_ds)

    if 'tail' in str(test_ratio):
        TestSelector = 'CaseRltLocInP'
        test_ratio = float(test_ratio.replace('tail', ''))
        test_threshold = 1 - test_ratio
    elif type(test_ratio) != float and type(test_ratio) != int:
        TestSelector = 'ObsDT'
        test_threshold = pd.to_datetime(test_ratio)
    else:
        TestSelector = 'RandTest'
        test_threshold = 1 - test_ratio

    if 'tail' in str(valid_ratio):
        ValidSelector = 'CaseRltLocInP'
        valid_ratio = float(valid_ratio.replace('tail', ''))
        valid_threshold = 1 - valid_ratio
    elif type(valid_ratio) != float and type(valid_ratio) != int:
        ValidSelector = 'ObsDT'
        valid_threshold = pd.to_datetime(valid_ratio)
    else:
        ValidSelector = 'RandValidation' 
        if type(test_ratio) == float or type(test_ratio) == int:
            valid_threshold = 1 - (valid_ratio / (1 - test_ratio))
        else: 
            valid_threshold = 1
        
    df_dsmp['Test'] = df_dsmp[TestSelector] >= test_threshold
    df_dsmp['Valid'] = (df_dsmp[ValidSelector] > valid_threshold) & (df_dsmp['Test'] == False)
    df_dsmp['Train'] = (df_dsmp['Test'] == False) & (df_dsmp['Valid'] == False)
    assert df_dsmp[['Train', 'Valid', 'Test']].sum(axis = 1).mean() == 1
    df_dsmp = df_dsmp.drop(columns = random_columns)
    return df_dsmp


def conduct_grouping_ds_case(df_case, groupname_columns, case_id_columns):
    if 'TVT' in groupname_columns:
        conditions = [
            df_case['Test'],
            df_case['Valid'],
            df_case['Train']
        ]
        choices = ['Test', 'Valid', 'Train']
        df_case['TVT'] = np.select(conditions, choices, default=np.nan)

    if 'InOut' in groupname_columns:
        conditions = [
            df_case['In'],
            df_case['Out'],
        ]
        choices = ['In', 'Out']
        df_case['InOut'] = np.select(conditions, choices, default=np.nan)
    
    if 'Base' in groupname_columns:
        df_case['Base'] = 'Base'

    df_case['group'] = df_case[groupname_columns].agg('_'.join, axis=1)

    groupname_to_dfcase = {}
    final_columns = case_id_columns
    for groupname, df in df_case.groupby('group'):
        groupname_to_dfcase[groupname] = datasets.Dataset.from_pandas(df[final_columns].reset_index(drop=True))
    return groupname_to_dfcase  

        
class Base:

    EXTERNAL_DATA_DICT = {}
    
    def __getstate__(self):
        # Return the state to be pickled, excluding the unpickleable attributes
        state = self.__dict__.copy()
        # Remove unpickleable entries
        if hasattr(self, 'dynamic_fn_names'):
            for fn_name in self.dynamic_fn_names:
                state[fn_name] = fn_name
        return state
    

    @staticmethod
    def sort_fn(s):
        try:
            return int(s.split(':')[-1].split('~')[0])
        except:
            return float('inf')
        
    @staticmethod
    def function_is_empty(func):
        source = inspect.getsource(func)
        # Remove any indentation
        if 'pass\n' in source:
            return True 
        else:
            return False 
        

    @staticmethod
    def load_module_variables(file_path):
        """Load a module from the given file path."""
        name = file_path
        spec = importlib.util.spec_from_file_location(name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    

    @staticmethod
    def convert_variables_to_pystirng(string_variables = None, 
                                      iterative_variables = None, 
                                      fn_variables = None, 
                                      prefix = None):
        
        if string_variables is None:
            string_variables = []
        if iterative_variables is None:
            iterative_variables = []
        if fn_variables is None:
            fn_variables = []
        if prefix is None:
            prefix = ['import pandas as pd', 'import numpy as np']

        L = prefix
        for i in string_variables:
            line = f'{retrieve_name(i)} = "{i}"'
            L.append(line)
            
        for i in iterative_variables:
            if type(i) == dict:
                pretty_str = pprint.pformat(i, 
                                            width=100, 
                                            sort_dicts=False, 
                                            compact=True,)
            else:
                pretty_str = str(i)
            # i = pretty_str
            line = f'{retrieve_name(i)} = {pretty_str}'
            L.append(line)
            
        for i in fn_variables:
            if i is None: continue
            line = f'{i.fn_string}'
            L.append(line)
            
        D_str = "\n\nMetaDict = {\n" + ',\n'.join(
                        ['\t' + f'"{retrieve_name(i)}": {retrieve_name(i)}'
                    for i in string_variables + iterative_variables + fn_variables]
                    ) + "\n}"
        
        python_strings = '\n\n'.join(L) + D_str
        
        return python_strings

    