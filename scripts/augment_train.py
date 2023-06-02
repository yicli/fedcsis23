from preprocess.feature_loader import FeatureDataset
import numpy as np
import pandas as pd

# features = [
#     'csv', 'SYSCALL_exit_isNeg', 'CUSTOM_openSockets_count',
#     'CUSTOM_openFiles_count', 'CUSTOM_libs_count', 'spawn_count'
# ]
# ds = FeatureDataset('train_local_scaled', features)
#
# class1 = ds.index.csv[ds.index.labels == 1]
# df1 = ds.features.loc[class1]
# df1.reset_index(inplace=True)
# df1.to_parquet('data/features/train_aug/class1.parquet')

df1 = pd.read_parquet('data/features/class1.parquet')

df1.set_index('csv', inplace=True)
df1 = df1.iloc[:, [0, 1, 2, 3, -3, -2, -1]]
df1 = df1.replace({0: np.nan})
std = df1.std()

df1_valid = df1.iloc[:70]
df1_train = df1.iloc[70:]

dfs = []
for i in range(1, 28):
    new = df1_valid + np.random.randn(*df1_valid.shape) * std.values
    new.index = new.index.astype(str) + '_' + str(i)
    dfs.append(new)

    # if i % 4 == 0:
    #     n = i/4
    #     save_dir = 'data/features/train_aug/class1_%i.parquet' % n
    #     shard = pd.concat(dfs)
    #     shard.clip(0, 1, inplace=True)
    #     shard.fillna(0, inplace=True)
    #     shard.reset_index(inplace=True)
    #     shard.to_parquet(save_dir, compression='gzip', engine='pyarrow')
    #     dfs = []

save_dir = 'data/features/train_aug/class1_valid.parquet'
shard = pd.concat(dfs + [df1_valid])
shard.clip(0, 1, inplace=True)
shard.fillna(0, inplace=True)
shard.reset_index(inplace=True)
shard.to_parquet(save_dir, compression='gzip', engine='pyarrow')
dfs = []