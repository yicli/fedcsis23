import logging
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
from torch.utils.data import DataLoader
from model.conv_net import ConvNet2Blk, ConvNet3Blk, ConvNet2BlkMP, ConvNetLogReg
from preprocess.feature_loader import FeatureDataset, collate_logs

logging.basicConfig(level=logging.INFO)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_sz = 64
# n_channels = [6, 8, 4]
run_name = 'conv_log_reg_nonres'
res = False
dataset_name = 'test_scaled'
# n_kernels = [10, 10, 10]
features = [
    'csv', 'SYSCALL_exit_isNeg', 'CUSTOM_openSockets_count',
    'CUSTOM_openFiles_count', 'CUSTOM_libs_count', 'spawn_count'
]

logging.info('Initialising model for inference...')
# model = ConvNet2BlkMP(n_channels, n_kernels, 'batch', residual=True)
model = ConvNetLogReg('batch', res)
checkpoint = torch.load(
    os.path.join('runs', run_name, 'checkpoint.tar')
)
model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()

logging.info('Initialising data loader...')
test_set = FeatureDataset(dataset_name, features)
test_loader = DataLoader(
    test_set, batch_size=batch_sz, shuffle=False,
    collate_fn=collate_logs
)

csvs = ()
preds = torch.tensor([]).to(device)
# ys = torch.tensor([]).to(device)
with torch.no_grad():
    for x, _, csv in iter(test_loader):
        x = x.to(device)
        # y = y.to(device)
        y_hat = model(x)
        csvs += csv
        preds = torch.cat((preds, y_hat))
        # ys = torch.cat((ys, y))

# pred_lab = preds > 0.5
# n_correct = (pred_lab == ys).sum()
# acc = n_correct / len(ys)
# print('Accuracy: %.3f' % acc)

result = pd.DataFrame({'pred': preds.cpu()}, index=csvs)
# test_order_file = os.path.join('data', 'test_files_ordering_for_submissions.txt')
# with open(test_order_file, 'r') as file:
#     labels = [line.rstrip() for line in file]
# test_order = pd.DataFrame({'order': labels})
# test_order = test_order.join(result, on='order')
# np.savetxt('inf' + run_name +'.txt', test_order.pred.values, fmt='%.10f')

# load y labels
test_label = pd.read_csv('data/test_labels.csv', header=None, index_col=0)
test_label.columns = ['label']
y_test = pd.concat([test_label, result], axis=1)

# calc AUC
score = roc_auc_score(y_test.label, y_test.pred)
print('Test AUC Score:', score)
