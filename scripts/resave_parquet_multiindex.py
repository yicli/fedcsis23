import os
from ast import literal_eval
import pandas as pd

# resave feature columns
fedcsis_dir = '/home/yichaoli8/fed_csis23/'
parquet = os.path.join(fedcsis_dir, 'features', 'train', 'shard0.parquet')
df = pd.read_parquet(parquet)
cols = pd.Series(df.columns)\
    .apply(literal_eval)\
    .to_list()
df.columns = pd.MultiIndex.from_tuples(cols)
df.to_parquet(parquet, engine='pyarrow', compression='gzip')
