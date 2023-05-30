import os
import pandas as pd
import numpy as np
from random import sample
import logging
from multiprocessing import Pool

columns_to_load = [
    'SYSCALL_timestamp', 'SYSCALL_pid', 'SYSCALL_syscall', 'SYSCALL_success',
    'SYSCALL_exit', 'PROCESS_comm', 'PROCESS_exe',
    'PROCESS_name', 'PROCESS_PATH', 'PROCESS_uid', 'PROCESS_gid',
    'CUSTOM_openFiles', 'CUSTOM_libs', 'CUSTOM_openSockets'
]
excluded_columns = [
    'SYSCALL_exit_hint', 'USER_AUTH', 'CRED_COUNT', 'SERVICE_COUNT', 'USER_ACTION_op',
    'USER_ACTION_src', 'USER_ACTION_res', 'USER_ACTION_addr'
]
columns_same_value = [
    'SYSCALL_arch', 'USER_MGMT_COUNT', 'USER_ERR_COUNT',
    'USYS_CONFIG_COUNT', 'CHID_COUNT', 'SELINUX_ERR_COUNT', 'SYSTEM_COUNT',
    'DAEMON_COUNT', 'NETFILTER_COUNT', 'SECCOMP_COUNT', 'AVC_COUNT',
    'ANOM_COUNT', 'INTEGRITY_COUNT', 'KERNEL_COUNT', 'RESP_COUNT',
    'SELINUX_MGMT_COUNT', 'KILL_process', 'KILL_uid'
]
subsumed_cols = ['PROCESS_comm', 'PROCESS_exe']
# Note: PROCESS_exe is subsumed by PROCESS_comm, however PROCESS_name sometimes
#   differ from PROCESS_exe, so PROCESS_exe is retained for comparison
#   SYSCALL_exit_hint acts as decode for SYSCALL_exit < 0, otw the same value

one_hot_cols = [
    'SYSCALL_syscall', 'SYSCALL_success', 'PROCESS_comm', 'PROCESS_exe',
    'PROCESS_PATH', 'PROCESS_uid', 'PROCESS_gid', 'PROCESS_name'
]


def load_paths(path_list):
    """Function used by multiprocessing to load a list file paths as pd.DataFrame"""
    assert hasattr(path_list, '__iter__')
    dfs = []
    for fpath in path_list:
        df = pd.read_csv(fpath, usecols=columns_to_load)
        fname = os.path.split(fpath)[-1] \
            .split('.')[0]
        df['csv'] = fname
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


class CsvLoader:
    def __init__(self, csv_dir):
        """
        :param csv_dir: where raw csv files are kept
        """
        self.csv_dir = csv_dir
        self.data_dir = {}
        self.data_files = {}

        # list train files
        self.data_dir['train'] = os.path.join(csv_dir, 'train_data')
        train_files = os.listdir(self.data_dir['train'])
        train_files.sort()
        self.data_files['train'] = pd.Series(train_files)

        # list test files
        self.data_dir['test'] = os.path.join(csv_dir, 'test_data')
        test_files = os.listdir(self.data_dir['test'])
        test_files.sort()
        self.data_files['test'] = pd.Series(test_files)

        # get train labels
        label_dir = os.path.join(csv_dir, 'train_files_containing_attacks.txt')
        with open(label_dir, 'r') as file:
            train_files_1 = [line.replace('\n', '.csv') for line in file]

        # identify train files where y=1
        mask_1 = [fname in train_files_1 for fname in train_files]
        self.mask_1 = pd.Series(mask_1)

        self.pool = Pool()

    def load_sample_csv(self, n_0=100, n_1=100):
        """
        randomly sample from both classes
        :param n_0: n samples from class 0
        :param n_1: n samples from class 1
        :return: pd.DataFrame
        """
        # random sample
        sample_0 = sample(list(self.data_files['train'][~self.mask_1]), n_0)
        sample_0 = [self.data_dir['train'] + '/' + fname for fname in sample_0]
        sample_1 = sample(list(self.data_files['train'][self.mask_1]), n_1)
        sample_1 = [self.data_dir['train'] + '/' + fname for fname in sample_1]

        # read in pd (selected cols)
        # sample_df0 = pd.concat([pd.read_csv(fname, usecols=columns_to_load) for fname in sample_0],
        #                        ignore_index=True)
        # sample_df1 = pd.concat([pd.read_csv(fname, usecols=columns_to_load) for fname in sample_1],
        #                        ignore_index=True)
        path_split0 = np.array_split(sample_0, self.pool._processes)
        sample_df0 = pd.concat(
            self.pool.map(load_paths, path_split0),
            ignore_index=True
        )
        path_split1 = np.array_split(sample_1, self.pool._processes)
        sample_df1 = pd.concat(
            self.pool.map(load_paths, path_split1),
            ignore_index=True
        )

        # Load all columns
        # sample_df0 = pd.concat([pd.read_csv('../train_data/' + fname) for fname in sample_0],
        #                        ignore_index=True)
        # sample_df1 = pd.concat([pd.read_csv('../train_data/' + fname) for fname in sample_1],
        #                        ignore_index=True)
        sample_df0['y'] = 0
        sample_df1['y'] = 1
        return pd.concat([sample_df0, sample_df1], ignore_index=True)

    def load_columns(self, train_test, columns: list[str], n_records=None):
        assert train_test in ('train', 'test')
        assert isinstance(columns, list)
        file_list = self.data_files[train_test][:n_records]
        file_paths = [os.path.join(self.data_dir[train_test], fname) for fname in file_list]
        logging.info('loading columns %s ...' % str(columns))
        return pd.concat([pd.read_csv(fpath, usecols=columns) for fpath in file_paths],
                         ignore_index=True)

    def load_row_range(self, train_test: object, row_range: slice, pool) -> object:
        assert train_test in ('train', 'test')
        assert isinstance(row_range, slice)
        file_list = self.data_files[train_test][row_range]
        file_paths = [os.path.join(self.data_dir[train_test], fname) for fname in file_list]
        logging.info('loading row range %s ...' % str(row_range))

        path_split = np.array_split(file_paths, pool._processes)
        dfs = pool.map(load_paths, path_split)
        return pd.concat(dfs, ignore_index=True)


class ParquetLoader:
    """Redundant, use pyarrow.parquet.ParquetDataset"""
    def __init__(self, parquet_dir):
        self.data_dir = parquet_dir
        file_list = os.listdir(self.data_dir)
        file_list.sort()
        self.data_files = pd.Series(file_list)

    def load_columns(self, columns: list[str], shard_list):
        assert isinstance(columns, list)
        file_list = self.data_files[shard_list]
        file_paths = [os.path.join(self.data_dir, fname) for fname in file_list]
        logging.info('loading columns %s ...' % str(columns))
        df = pd.concat([pd.read_parquet(fpath, columns=columns) for fpath in file_paths],
                       ignore_index=True)
        logging.info('done')
        return df


if __name__ == '__main__':
    loader = CsvLoader('/home/yichaoli8/fed_csis23')
    df = loader.load_columns(['SYSCALL_syscall'], n_records=10000)
