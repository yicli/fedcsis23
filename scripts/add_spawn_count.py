from util.util import get_columns_by_lv0
from preprocess.features import PdApplyPool
import os
import pyarrow.parquet as pa
import pandas as pd
from sklearn.preprocessing import minmax_scale

pqt_path = os.path.join('data', 'features', 'train_local_scaled', 'shard0.parquet')
pqt_file = pa.ParquetFile(pqt_path)
sc_cols = ['csv', 'SYSCALL_pid', 'SYSCALL_exit']
# sc_cols_multi = get_columns_by_lv0(pqt_file.schema.names, sc_cols)

df = pd.read_parquet(pqt_path)

p = PdApplyPool()
spc = p.spawn_count(df)
p.close_pool()
assert (df.csv == spc.index).all()
spc_scaled = minmax_scale(spc)

df['spawn_count'] = spc_scaled
df.to_parquet(pqt_path, engine='pyarrow', compression='gzip')
