from icu_experiments.load_data import load_data_for_prediction
import os
import pickle


def load_data(outcome):
    # Check if data has already been processed 
    outcome_data_path = outcome + '_data.pkl'
    if os.path.exists(outcome_data_path):
        print(f'The data file exists!')
        with open(outcome_data_path, 'rb') as data: 
            _data = pickle.load(data)
    else:
        _data = load_data_for_prediction(sources,  outcome=outcome, log_transform=True)
        with open(outcome_data_path, 'wb') as data:
            pickle.dump(_data, data)
    print(f'Data loaded successfully: {outcome_data_path}\n')
    return _data


def results_exist(path):
    if os.path.exists(path):
        print(f'The result file exists!')
        return True
    return False


def save_data(path,results): 
    with open(path, 'wb') as data:
        pickle.dump(results, data)
    print(f'Results stored successful')