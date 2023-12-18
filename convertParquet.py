import pandas as pd
import pickle

file='ridge_results'

with open(f'Pickle//{file}.pkl','rb') as data:
    _data=pickle.load(data)

df=pd.DataFrame(_data)

print(df)

df.to_parquet(f'Parquet//{file}.parquet',engine='pyarrow')