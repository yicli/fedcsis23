import os
import pandas as pd
import numpy as np

f_list = []
dir = 'data/features/train_aug'
for f in os.listdir(dir):
    if f.startswith('class1'):
        f_list.append(f)

csv_list = []
path_list = [os.path.join(dir, f) for f in f_list]
path_list += ['data/features/valid_aug/class1_valid.parquet']
for p in path_list:
    csv_list.extend(
        pd.read_parquet(p, columns=["('csv', '')"])
            .csv.unique()
            .tolist()
    )
label_dir = os.path.join('data', 'orig_train_files_containing_attacks.txt')
with open(label_dir, 'r') as file:
    labels = [line.rstrip() for line in file]

csv_list += labels

with open('data/train_files_containing_attacks.txt', 'w') as outfile:
    for line in csv_list:
        outfile.write(line + '\n')
