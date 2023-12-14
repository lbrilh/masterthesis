from icu_experiments.load_data import load_data_for_prediction
from set_up import sources
import os
import glob
import pandas as pd


'''def load_data(outcome):
    current_directory = os.getcwd()
    relative_path = os.path.join('Parquet', f'{outcome}_data.parquet')
    file_path = os.path.join(current_directory, relative_path)
    if os.path.exists(file_path):
        print(f'The data file exists!')
        _data = pd.read_parquet(file_path, engine='fastparquet')
    else:
        _data = load_data_for_prediction(sources,  outcome=outcome, log_transform=True)
        data = pd.DataFrame(_data)
        data.to_parquet(f'{outcome}_data.parquet')
    print(f'Data loaded successfully: {file_path}\n')
    return _data'''


def results_exist(path):
    current_directory = os.getcwd()
    relative_path = os.path.join('Parquet',path)
    file_path = os.path.join(current_directory, relative_path)
    if os.path.exists(file_path):
        print(f'The result file exists!')
        return True
    return False


def save_data(path,results): 
    current_directory = os.getcwd()
    relative_path = os.path.join('Parquet',path)
    file_path = os.path.join(current_directory, relative_path)
    data = pd.DataFrame(results)
    data.to_parquet(path)
    print(f'Results stored successful')


def retrieve_paths(pattern):
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, 'Parquet')
    file_paths = glob.glob(os.path.join(file_path, pattern))
    return file_paths


def load_data_plotting(model=None,path=None):
    if path:
        file_path=path
    else: 
        current_directory = os.getcwd()
        relative_path = os.path.join('Parquet', f'{model}_results.parquet')
        file_path = os.path.join(current_directory, relative_path)
        _data=pd.read_parquet(file_path, engine='fastparquet')
    return _data


'''
def load_data(outcome):
    current_directory = os.getcwd()
    relative_path = os.path.join('Pickle', f'{outcome}_data.pkl')
    file_path = os.path.join(current_directory, relative_path)
    if os.path.exists(file_path):
        print(f'The data file exists!')
        with open(file_path, 'rb') as data: 
            _data = pickle.load(data)
    else:
        _data = load_data_for_prediction(sources,  outcome=outcome, log_transform=True)
        with open(file_path, 'wb') as data:
            pickle.dump(_data, data)
    print(f'Data loaded successfully: {file_path}\n')
    return _data


def results_exist(path):
    current_directory = os.getcwd()
    relative_path = os.path.join('Pickle',path)
    file_path = os.path.join(current_directory, relative_path)
    if os.path.exists(file_path):
        print(f'The result file exists!')
        return True
    return False


def save_data(path,results): 
    current_directory = os.getcwd()
    relative_path = os.path.join('Pickle',path)
    file_path = os.path.join(current_directory, relative_path)
    with open(file_path, 'wb') as data:
        pickle.dump(results, data)
    print(f'Results stored successful')


def retrieve_paths(pattern):
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, 'Pickle')
    file_paths = glob.glob(os.path.join(file_path, pattern))
    return file_paths


def load_data_plotting(model=None,path=None):
    if path:
        file_path=path
    else: 
        current_directory = os.getcwd()
        relative_path = os.path.join('Pickle', f'{model}_results.pkl')
        file_path = os.path.join(current_directory, relative_path)
    with open(file_path, 'rb') as data:
        _data=pickle.load(data)
    return _data'''