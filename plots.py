import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from model.conv_net import ConvNet2Blk, ConvNet3Blk, ConvNet2BlkMP
from preprocess.feature_loader import FeatureDataset, collate_logs


def sum_keys(sd_list, key):
    n = len(sd_list)
    x = sd_list[0][key]
    for i in range(1, n):
        x += sd_list[i][key]
    return x


logging.basicConfig(level=logging.INFO)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_sz = 64
n_channels = [6, 8, 4]
run_name = 'aug2_ker101010mp'
dataset_name = 'test_scaled'
n_kernels = [10, 10, 10]
features = [
    'csv', 'SYSCALL_exit_isNeg', 'CUSTOM_openSockets_count',
    'CUSTOM_openFiles_count', 'CUSTOM_libs_count', 'spawn_count'
]
test_set = FeatureDataset(dataset_name, features)
#checkpoint_list = [24, 29, 34, 49, 54, 59, 64]
checkpoint_list = [24, 29, 34]
# checkpoint_list = [49, 54, 59, 64]

logging.info('Initialising model for inference...')
model = ConvNet2BlkMP(n_channels, n_kernels, 'batch', residual=True)

checkpoints = ['checkpoint%i.tar' % i for i in checkpoint_list]
checkpoints = [
    torch.load(os.path.join('runs', run_name, c), map_location=torch.device('cpu')) for c in checkpoints
]
state_dicts = [c['model_state'] for c in checkpoints]
avg_state = state_dicts[0]
for key in avg_state:
    avg_state[key] = sum_keys(state_dicts, key)/len(state_dicts)

l1ylabels = test_set[0][0].columns
l1 = avg_state['net.0.conv.weight']
ax = sns.heatmap(l1[2], linewidth=0.5, cmap=['green', 'white', 'red'], center=0, yticklabels=l1ylabels)
ax = sns.heatmap(l1.abs().sum(dim=0), cmap=['white', 'orange', 'red'], linewidth=0.5, yticklabels=l1ylabels)
plt.show()