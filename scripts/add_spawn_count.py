from util.util import get_columns_by_lv0
from preprocess.features import PdApplyPool
import os
import pyarrow.parquet as pa
import pandas as pd
from sklearn.preprocessing import minmax_scale
import logging
import gc

logging.basicConfig(level=logging.INFO)
p = PdApplyPool()

# for i in range(1, 8):
i = 0
fname = 'shard%i.parquet' % i
pqt_path = os.path.join('data', 'features', 'train_local_scaled', fname)
pqt_file = pa.ParquetFile(pqt_path)
# sc_cols = ['csv', 'SYSCALL_pid', 'SYSCALL_exit']
# sc_cols_multi = get_columns_by_lv0(pqt_file.schema.names, sc_cols)

df = pd.read_parquet(pqt_path)
spc = p.spawn_count(df)
assert (df.csv == spc.index).all()

spc_scaled = minmax_scale(spc)
# df['spawn_count'] = spc_scaled

df['spawn_count'] = spc.values
df.to_parquet(pqt_path, engine='pyarrow', compression='gzip')

del df, spc
gc.collect()

p.close_pool()
