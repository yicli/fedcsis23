import pandas as pd
from preprocess.data_loader import CsvLoader, columns_same_value

loader = CsvLoader('/home/yichaoli8/fed_csis23')
sample_df = loader.load_sample_csv(200, 200)

# sample_df[['CUSTOM_openSockets', 'y']].value_counts().unstack().plot()

# sample_df.iloc[:, 1].unique()  # allways aarch64
# sample_df.iloc[:, 2].unique()  # factorise
# sample_df.iloc[:, 3].unique()  # factorise

# count number of unique values per column
n_uniq = sample_df.apply(lambda col: len(col.unique()), axis=0)

# check if current sample has the same columns with only 1 value
columns_count_1 = (n_uniq[n_uniq == 1]).index
print((columns_count_1 == columns_same_value).all())
print(len(columns_count_1), len(columns_same_value))
print(len(sample_df))

#sample_df.PROCESS_comm + sample_df.PROCESS_exe
cols = ['PROCESS_comm', 'PROCESS_exe', 'PROCESS_name']
temp_df = sample_df[cols].copy()
temp_df['concat'] = temp_df.iloc[:, 0] + temp_df.iloc[:, 1] + temp_df.iloc[:, 2]
len(temp_df.iloc[:, 0].unique())
len(temp_df['concat'].unique())
temp_uniq = temp_df.drop_duplicates().sort_values(by='concat')

cols = ['SYSCALL_exit_hint', 'SYSCALL_exit']
temp_df = sample_df[cols].copy()
temp_df['SYSCALL_exit_hint_num'] = pd.to_numeric(sample_df.SYSCALL_exit_hint, errors='coerce')
# temp_df[temp_df.SYSCALL_exit_hint_num.isna()]
temp_df = temp_df.dropna(how='all')
temp_df['dif'] = temp_df.SYSCALL_exit_hint_num - temp_df.SYSCALL_exit
temp_df[temp_df.dif != 0].apply(lambda col: len(col.unique()), axis=0)
temp_df.SYSCALL_exit_hint.value_counts()
temp_df.SYSCALL_exit.value_counts()

case1 = pd.read_csv('../train_data/ffe735a6-0371-401d-9419-e025ddb4afd9.csv')
