import logging
import os
import pickle

import pandas as pd
import pyarrow.parquet as pa
from util.util import get_columns_by_lv0
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)
count_cols = ['CUSTOM_openSockets_count', 'CUSTOM_openFiles_count', 'CUSTOM_libs_count']
fedcsis_dir = '/home/yichaoli8/fed_csis23/'
scaler_write_path = os.path.join(fedcsis_dir, 'features', 'minmaxscaler', 'custom_count_scaler.pkl')


def create_scaler():
    feature_dir = os.path.join(fedcsis_dir, 'features', 'train_local')
    pds = pa.ParquetDataset(feature_dir)
    cols_to_load = get_columns_by_lv0(pds.schema.names, count_cols)
    df = pds.read_pandas(columns=cols_to_load) \
        .to_pandas()

    scaler = MinMaxScaler()
    scaler.fit(df)

    payload = {
        'feature_order': df.columns,
        'scaler': scaler
    }

    with open(scaler_write_path, 'wb') as f:
        pickle.dump(payload, f)


parquet_file = os.path.join(fedcsis_dir, 'features', 'train_local', 'shard0.parquet')
parquet_write_dir = os.path.join(fedcsis_dir, 'features', 'train_local_scaled', 'shard0.parquet')
df = pd.read_parquet(parquet_file)

with open(scaler_write_path, 'rb') as f:
    cols, scaler = pickle.load(f).values()

df.loc[:, cols] = scaler.transform(df.loc[:, cols])
df.to_parquet(parquet_write_dir, engine='pyarrow', compression='gzip')
