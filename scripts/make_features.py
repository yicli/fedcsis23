import logging
import gc
import os
import pandas as pd
from preprocess.features import Factorise, PdApplyPool, is_less_than_zero
from preprocess.data_loader import one_hot_cols, ParquetLoader

logging.basicConfig(level=logging.INFO)
data_set = 'test'
fedcsis_dir = '/home/yichaoli8/fed_csis23/'
# shard_name = 'shard0.parquet'
for i in range(1, 3):
    raw_data_dir = os.path.join(fedcsis_dir, 'shards', data_set)
    loader = ParquetLoader(raw_data_dir)
    shard_list = [i]
    shard_name = 'shard%i.parquet' % i

    # columns to use as is
    cols = ['csv', 'SYSCALL_timestamp', 'SYSCALL_pid', 'SYSCALL_exit']
    data = loader.load_columns(cols, shard_list=shard_list)
    features = data.csv.astype('category')
    features = pd.concat([
        features,
        data['SYSCALL_timestamp'].astype('uint16'),
        data['SYSCALL_pid'].astype('uint32'),
        data['SYSCALL_exit']
    ],
        axis=1
    )
    features.columns = pd.MultiIndex.from_product([features.columns, ['']])

    # exit code
    f = is_less_than_zero(data['SYSCALL_exit'])
    features = pd.concat([features, f], axis=1)

    del data
    gc.collect()

    # make features from CUSTOM cols
    cols = ['CUSTOM_openSockets', 'CUSTOM_openFiles', 'CUSTOM_libs']
    data = loader.load_columns(cols, shard_list=shard_list)
    pd_apply = PdApplyPool()
    f = pd_apply.custom_socket_fam_x_dir(data['CUSTOM_openSockets'])
    features = pd.concat([features, f], axis=1)

    for c in cols[1:]:
        f = pd_apply.custom_count(data[c])
        features = pd.concat([features, f], axis=1)

    features.fillna(0, inplace=True)  # Note: exit code is also zero filled!

    dtype = pd.Series(
        ['category', 'uint16', 'uint32', 'float64', 'uint8'] + ['uint8'] * 12,
        index=features.columns
    )
    features = features.astype(dtype)

    del data
    gc.collect()

    cols = ['csv', 'SYSCALL_pid', 'SYSCALL_exit']
    data = loader.load_columns(cols, shard_list=shard_list)

    spc = pd_apply.spawn_count(data)
    pd_apply.close_pool()
    assert (features.csv == spc.index).all()
    features['spawn_count'] = spc.values

    del data, pd_apply
    gc.collect()

    # one hot features
    # for c in one_hot_cols:
    #     data = loader.load_columns([c], shard_list=shard_list)
    #     fact = Factorise(c)
    #     fact.load_encoder()
    #     f = fact.transform(data[[c]])
    #     f.columns = pd.MultiIndex.from_product([[c], f.columns])
    #     features = pd.concat([features, f], axis=1)
    #
    #     del data, fact, f
    #     gc.collect()

    write_path = os.path.join('../data', 'features', data_set, shard_name)
    features.to_parquet(write_path, engine='pyarrow', compression='gzip')
