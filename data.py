from icu_experiments.load_data import load_data_for_prediction
from set_up import sources
import os
import pickle
import glob


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
    relative_path = os.path.join('Pickle',path)
    file_path = os.path.join(current_directory, relative_path)
    file_paths = glob.glob(os.path.join(file_path, pattern))
    return file_paths


def load_data_plotting(model):
    current_directory = os.getcwd()
    relative_path = os.path.join('Pickle', f'{model}_results.pkl')
    file_path = os.path.join(current_directory, relative_path)
    with open(file_path, 'rb') as data:
        _data=pickle.load(data)
    return _data