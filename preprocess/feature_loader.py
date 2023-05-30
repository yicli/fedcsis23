import os
import pandas as pd
import pyarrow.parquet as pa
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import pad

from util.util import get_columns_by_lv0
# from preprocess.data_loader import ParquetLoader


class FeatureDataset(Dataset):
    def __init__(self, feature_set, feature_list):
        if 'csv' not in feature_list:
            feature_list.append('csv')
        feature_dir = os.path.join('/home/yichaoli8/fed_csis23/features', feature_set)
        parquet_ds = pa.ParquetDataset(feature_dir)
        cols_to_load = get_columns_by_lv0(parquet_ds.schema.names, feature_list)
        self.features = parquet_ds.read_pandas(columns=cols_to_load) \
            .to_pandas()
        self.features.set_index(('csv', ''), inplace=True)
        self.index = pd.DataFrame({'csv': self.features.index.unique()})

        label_dir = '/home/yichaoli8/fed_csis23/train_files_containing_attacks.txt'
        with open(label_dir, 'r') as file:
            labels = [line.rstrip() for line in file]
        self.index['labels'] = self.index.csv.isin(labels).astype('uint8')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        csv = self.index.loc[item, 'csv']
        x = self.features.loc[csv]\
            .astype('float32')
        y = self.index.loc[item, 'labels']\
            .astype('float32')
        return torch.tensor(x.values), torch.tensor(y)


def collate_logs(item_list: list[tuple]):
    x, y = list(zip(*item_list))
    x_dim0 = [e.shape[0] for e in x]
    dim0 = max(x_dim0)
    pad_dim0 = [(0, 0, 0, dim0 - e) for e in x_dim0]
    _zip = zip(x, pad_dim0)
    x = torch.stack([pad(_x, _pad) for _x, _pad in _zip])
    x = torch.swapaxes(x, 1, 2)     # treat feature dim as channel
    y = torch.stack(y)
    return x, y


if __name__ == '__main__':
    ds = FeatureDataset('train_local_scaled', ['csv', 'SYSCALL_syscall', 'PROCESS_comm'])
    items = [ds.__getitem__(i) for i in range(5)]
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_logs)
    out = next(iter(dl))
