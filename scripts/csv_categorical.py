import os
import pandas as pd
import numpy as np
import pickle

dir1 = 'data/features/train_aug2'
dir2 = 'data/features/valid_aug2'
path1 = [os.path.join(dir1, f) for f in os.listdir(dir1)]
path2 = [os.path.join(dir2, f) for f in os.listdir(dir2)]
paths = path1 + path2

# compile unique csv as np.array
csv_uniq = pd.DataFrame([])
for p in paths:
    csv = pd.read_parquet(p, columns=["('csv', '')"])\
        .csv.unique()
    csv = pd.DataFrame({'csv': csv})
    csv['file'] = p
    csv_uniq = pd.concat([csv_uniq, csv])

uniq_vals = csv_uniq.csv.values
# with open('data/features/csv_cats.pkl', 'wb') as f:
#     pickle.dump(uniq_vals, f)

# coerce to categorical
for p in paths:
    print(p)
    df = pd.read_parquet(p)
    df.loc[:, ('csv', '')] = pd.Categorical(df.csv, categories=uniq_vals)
    df.to_parquet(p, engine='pyarrow', compression='gzip')

