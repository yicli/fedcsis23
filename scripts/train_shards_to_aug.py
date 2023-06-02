import pandas as pd
import os
import pyarrow.parquet as pa

in_path = 'data/features/train_scaled'
out_path = 'data/features/train_aug'

cols = pa.ParquetFile(
    os.path.join(out_path, 'class1_0.parquet')
).schema.names

for f in os.listdir(in_path):
    df = pd.read_parquet(
        os.path.join(in_path, f),
        columns=cols
    )
    df.loc[:, ('csv', '')] = df['csv'].astype(str)
    df.to_parquet(
        os.path.join(out_path, f),
        engine='pyarrow',
        compression='gzip'
    )

# remove class1
label_dir = os.path.join('data', 'orig_train_files_containing_attacks.txt')
with open(label_dir, 'r') as file:
    labels = [line.rstrip() for line in file]

f_list = []
dir = 'data/features/train_aug'
for f in os.listdir(dir):
    if f.startswith('shard'):
        f_list.append(f)
path_list = [os.path.join(dir, f) for f in f_list]

for p in path_list:
    print(p)
    df = pd.read_parquet(p)
    l1 = len(df)
    print('df len', l1)
    mask = ~df.csv.isin(labels)
    df = df[mask]
    l2 = len(df)
    print('new df len', l2)
    print('removed lines', 1 - l2 / l1)
    print('write parquet')
    df.to_parquet(p, engine='pyarrow', compression='gzip')

