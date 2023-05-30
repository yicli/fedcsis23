from scipy.spatial import distance_matrix
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import networkx as nx
import os
import pickle
import logging
from ast import literal_eval
from multiprocessing import Pool
from numpy import array_split


# Relationship Matrices: time, exit pid, pid
def time_matrix(window: pd.DataFrame):
    timestamp = window[['SYSCALL_timestamp']]
    return distance_matrix(timestamp, timestamp)


def pid_matrix(window: pd.DataFrame):
    pid = window[['SYSCALL_pid']]
    dist = distance_matrix(pid, pid)
    return (dist == 0).astype(int)


def exit_matrix(window: pd.DataFrame):
    exit_code = window[['SYSCALL_exit']] \
        .reset_index() \
        .set_index('SYSCALL_exit')
    pid = window[['SYSCALL_pid']] \
        .reset_index() \
        .set_index('SYSCALL_pid')
    edges = exit_code \
        .join(pid, how='left', lsuffix='_from', rsuffix='_to') \
        .dropna(how='any') \
        .astype(int) \
        .values
    dg = nx.DiGraph()
    n_nodes = len(window)
    dg.add_nodes_from(range(0, n_nodes))
    dg.add_edges_from(edges)
    return nx.adjacency_matrix(dg).todense()


class SkLearnFeature:
    def __init__(self, feat_name):
        self.enc = None
        self.feat_name = feat_name
        self.feat_dir = '/home/yichaoli8/fed_csis23/features/'

        # date_str = datetime.today().strftime('%y%m%d')
        # self.fname = self.feat_name + date_str
        self.fname = self.feat_name

    def write_encoder(self):
        write_path = os.path.join(self.feat_dir, 'encoders', self.fname + '.pkl')
        logging.info('writing encoder to %s ...' % write_path)
        with open(write_path, 'wb') as f:
            pickle.dump(self.enc, f)
        logging.info('done')

    def load_encoder(self, load_path=None):
        if not load_path:
            load_path = os.path.join(self.feat_dir, 'encoders', self.fname + '.pkl')
        with open(load_path, 'rb') as f:
            self.enc = pickle.load(f)

    def write_feature(self, feat_df, train_test):
        assert train_test in ('train', 'test')
        write_path = os.path.join(self.feat_dir, train_test, self.fname + '.parquet')
        logging.info('converting DataFrame to dense...')
        dense = feat_df.sparse.to_dense()
        logging.info('writing to file %s ...' % write_path)
        dense.to_parquet(write_path, engine='pyarrow', compression='gzip', index=False)
        logging.info('done')

    def write_train_artefacts(self, feat_df):
        self.write_encoder()
        self.write_feature(feat_df, 'train')


class Factorise(SkLearnFeature):
    def __init__(self, feat_name):
        super().__init__(feat_name)

    def fit(self, data):
        self.enc = OneHotEncoder(dtype=int, handle_unknown='ignore')
        self.enc.fit(data)

    def transform(self, data):
        logging.info('one-hot encoding column %s ...' % self.feat_name)
        spmatrix = self.enc.transform(data).todense()
        logging.info('done')
        return pd.DataFrame(
            spmatrix,
            index=data.index,
            columns=self.enc.categories_[0],
            dtype='uint8'
        )


def is_less_than_zero(data):
    res = (data < 0).astype('uint8')
    res.name = data.name + '_isNeg'
    res = pd.DataFrame(res)
    res.columns = pd.MultiIndex.from_product([res.columns, ['']])
    return res


class PdApplyPool:
    def __init__(self):
        self.pool = Pool()

    def send_to_pool(self, func, data):
        data_split = array_split(data, self.pool._processes)
        return pd.concat(self.pool.map(func, data_split))

    def close_pool(self):
        self.pool.close()
        self.pool.join()

    def custom_socket_fam_x_dir(self, data: pd.Series):
        assert isinstance(data, pd.Series)
        data = data[data != '[]']
        # res = self.pool.send_to_pool(worker_literal_eval, data)
        logging.info('send to pool: count open socket family X direction ...')
        res = self.send_to_pool(worker_family_direction_count, data)
        logging.info('done')
        col_name_merged = res.columns.map('_'.join)
        feat_name = data.name + '_count'
        res.columns = pd.MultiIndex.from_product([[feat_name], col_name_merged])
        return res

    def custom_count(self, data: pd.Series):
        assert isinstance(data, pd.Series)
        data = data[data != '[]']
        logging.info('send to pool: len literal_eval: %s' % data.name)
        res = self.send_to_pool(worker_literal_eval_len, data)
        logging.info('done')
        res.name = data.name + '_count'
        res = pd.DataFrame(res)
        res.columns = pd.MultiIndex.from_product([res.columns, ['']])
        return res


def worker_literal_eval(split):
    return split.apply(literal_eval)


def worker_literal_eval_len(split):
    return split.apply(
        lambda x: len(literal_eval(x))
    )


def worker_family_direction_count(split):
    s = split.apply(
        lambda x: pd.Series(literal_eval(x))
    )\
        .stack()
    df = pd.DataFrame(list(s), index=s.index)
    gb = df.groupby(df.index.get_level_values(0))
    out = gb[['family', 'direction']] \
        .value_counts() \
        .unstack(level=['family', 'direction'])
    return out


def worker_df_value_counts(split):
    fd = split.apply(
        lambda x: pd.DataFrame(x)[['family', 'direction']].value_counts()
    )
    return fd


#TODO: Mean number of Custom Open Files is similar for both classes,
# but are there open files that are more associated with 1?
# Same question for CUSTOM_libs; use ast.literal_eval(x)

if __name__ == '__main__':
    # investigation on 'PROCESS_comm', 'PROCESS_exe', 'PROCESS_name' ---- inconclusive
    import pyarrow.parquet as pa

    ds = pa.ParquetDataset('../shards/train_local/')
    col_names = ['csv', 'PROCESS_comm', 'PROCESS_exe', 'PROCESS_name']

    # col_names = get_columns_by_lv0(ds.schema.names, col_names)
    df = ds.read_pandas(columns=col_names).to_pandas()

    label_dir = '../train_files_containing_attacks.txt'
    with open(label_dir, 'r') as file:
        labels = [line.rstrip() for line in file]
    df['y'] = df.csv.isin(labels)

    df['exe_ne_name'] = (df.PROCESS_name != df.PROCESS_exe)
    df['exe_last'] = df.PROCESS_exe.str.split('/').str[-1]
    df['name_last'] = df.PROCESS_name.str.split('/').str[-1]

    df['all_equal'] = (df['PROCESS_comm'] == df['name_last']) & (~df.exe_ne_name)