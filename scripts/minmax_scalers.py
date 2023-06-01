import logging
import os
import pickle

import pandas as pd
import pyarrow.parquet as pa
from util.util import get_columns_by_lv0
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)
count_cols = ['CUSTOM_openSockets_count', 'CUSTOM_openFiles_count', 'CUSTOM_libs_count']
feature_dir = os.path.join('../data', 'features')
scaler_dir = os.path.join(feature_dir, 'minmaxscaler')


def create_scaler(cols_to_scale, scaler_name, dataset='train'):
    pqt_dir = os.path.join(feature_dir, dataset)
    pds = pa.ParquetDataset(pqt_dir)
    cols_to_load = get_columns_by_lv0(pds.schema.names, cols_to_scale)
    df = pds.read_pandas(columns=cols_to_load) \
        .to_pandas()

    scaler = MinMaxScaler()
    scaler.fit(df)

    payload = {
        'feature_order': df.columns,
        'scaler': scaler
    }

    scaler_write_path = os.path.join(scaler_dir, scaler_name+'.pkl')
    with open(scaler_write_path, 'wb') as f:
        pickle.dump(payload, f)

    return scaler


# scale and save parquet
def scale(dataset, shard):
    fname = 'shard%i.parquet' % shard
    parquet_file = os.path.join(feature_dir, dataset, fname)
    parquet_write_dir = os.path.join(feature_dir, dataset+'_scaled', fname)
    df = pd.read_parquet(parquet_file)

    for s in ['custom_count.pkl', 'spawn_count.pkl']:
        path = os.path.join(scaler_dir, s)
        with open(path, 'rb') as f:
            cols, scaler = pickle.load(f).values()
        df.loc[:, cols] = scaler.transform(df.loc[:, cols])

    df.to_parquet(parquet_write_dir, engine='pyarrow', compression='gzip')


if __name__ == '__main__':
    # spawn_scaler = create_scaler(['spawn_count'], 'spawn_count_local', 'train_local')

    # for shard in range(3):
    #     scale('test', shard)

    for shard in range(8):
        logging.info('Processing shard %i' % shard)
        scale('train', shard)
