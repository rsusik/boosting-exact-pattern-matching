from pers import PersistentResults
import pandas as pd
from collections import Counter
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
tqdm.pandas()



# Config
algorithms = ['bf', 'kr', 'qs', 'nsn', 'smith', 'rcolussi', 'askip', 'br', 'fs', 'ffs', 'bfs', 'ts', 'ssabs', 'hash3', 'hash5', 'hash8', 'aut', 'rf', 'bom', 'bom2', 'ildm1', 'ildm2', 'ebom', 'fbom', 'sebom', 'sfbom', 'so', 'sa', 'bndm', 'bndml', 'sbndm', 'sbndm2', 'sbndm-bmh', 'faoso2', 'aoso2', 'aoso4', 'fsbndm', 'bndmq2', 'bndmq4', 'bndmq6', 'sbndmq2', 'sbndmq4', 'sbndmq6', 'sbndmq8', 'ufndmq4', 'ufndmq6', 'ufndmq8']

alg_name_to_id = {}
for idx, alg in enumerate(algorithms):
    alg_name_to_id[alg] = idx

results = PersistentResults(
    'dataset-raw.pickle',
)
df = pd.DataFrame(results.data)

algorithms = list(set(algorithms).intersection(set(df['algorithm'])))

df = df[df['algorithm'].isin(algorithms)]

print(sorted(df['algorithm'].unique()))

if 'isException' in df.columns:
    df2 = df[df['isException'] != True].copy()
else:
    df2 = df.copy()

max_m = max(df2['m'].unique())
print(max_m)


# %% # Remove bad data

ref_occ = defaultdict(dict)

for (ds, p), val in df[df['algorithm']=='bf'].groupby(['dataset', 'pattern'])['occurences']:
    ref_occ[ds][p] = val.values[0]

def check(x):
    occ = ref_occ[x['dataset']][x['pattern']]
    return x['occurences'] == occ

tmp = df2[
    (df2['pattern'].str.len() == df2['m']) & 
    (df2['total_time']!=0) & 
    (df2['run_time']!=0)
].copy()[['pattern', 'dataset', 'm', 'n', 'algorithm', 'total_time']]

assert len(df2) - len(tmp) == 0, 'cos poszlo nie tak'

df2 = tmp



# %%
# Dataset split - 500 train, 500 test

df_bf = df[df['algorithm'] == 'bf']
df_bf_filtered = (df_bf.groupby('pattern').count() > 1)['m'].reset_index()
df_duplicates = df_bf_filtered[df_bf_filtered['m'] == True]
duplicates = df_duplicates['pattern'].to_list()

random.seed(123)
test_patterns = []
train_patterns = []


train_df = pd.DataFrame()
test_df = pd.DataFrame()
for dataset, group in tqdm(df2.groupby('dataset')):
    for m, group2 in group.groupby('m'):
        uniq_patterns = group2['pattern'].unique()
        assert len(uniq_patterns)==1000, 'Cos poszlo nie tak'
        random.shuffle(uniq_patterns)
        train_patterns = uniq_patterns[:500].tolist()
        test_patterns = uniq_patterns[500:].tolist()
        # if there is a duplicate in test set, then swap it with non-duplicate from train set
        # to ensure there are no duplicate entries shared by test and train sets
        list_to_move_to_train = []
        for el in test_patterns:
            if el in duplicates:
                # duplicates from test set are added to list to move to train set
                print('found duplicate in test set, moving to train set')
                list_to_move_to_train.append(el)
        for el in list_to_move_to_train:
            # remove duplicates from test set and move to train set
            test_patterns.remove(el)
            train_patterns.append(el)
            for el2 in train_patterns:
                if (
                    el2 not in duplicates and 
                    el2 not in test_patterns
                ):
                    # next, find element in train set which is not a duplicate and move it to test set
                    print('found not duplicate element in train set, moving to test set')
                    test_patterns.append(el2)
                    train_patterns.remove(el2)
                    break
            print()

        assert len(train_patterns) == 500, 'len(train_patterns) != 500'
        assert len(test_patterns) == 500, 'len(test_patterns) != 500'


        train_records = df[df['pattern'].isin(train_patterns)&(df['dataset']==dataset)&(df['m']==m)]
        train_df = pd.concat([train_df, train_records])
        test_records = df[df['pattern'].isin(test_patterns)&(df['dataset']==dataset)&(df['m']==m)]
        test_df = pd.concat([test_df, test_records])

assert len(set(test_patterns).intersection(set(train_patterns))) == 0, 'test i train maja elementy wspolne'

ttr = train_df.groupby(['dataset', 'm']).count()[train_df.groupby(['dataset', 'm']).count().columns[0]]
tts = test_df.groupby(['dataset', 'm']).count()[test_df.groupby(['dataset', 'm']).count().columns[0]]

error_occured = False
for x in zip(list(zip(ttr.to_dict().keys(), ttr.to_dict().values())), list(zip(tts.to_dict().keys(), tts.to_dict().values()))):
    if x[0][1] != x[1][1]:
        error_occured = True
        print(x[0][0], x[0][1], x[1][1])

if error_occured:
    print('ERROR: Some groups have different number of elements in train and test sets')
    exit(1)


# %%
# Best times

from datautils import get_timings

cols = ['algorithm', 'pattern', 'dataset', 'occurences', 'm', 't', 'n']

train_time = get_timings(train_df, cols) # dane do klasyfikacji czasow (nie klas)
test_time = get_timings(test_df, cols) # dane do klasyfikacji czasow (nie klas)

train_time['algorithm'] = train_time[algorithms].idxmin(axis=1)
train_time = train_time[cols + algorithms]

test_time['algorithm'] = test_time[algorithms].idxmin(axis=1)
test_time = test_time[cols + algorithms]


# %% Y

y_train_time = train_time[algorithms].to_numpy()
y_test_time  = test_time[algorithms].to_numpy()

algos = list(set(train_time.columns).intersection(algorithms))

print('Algorithms that won at least once')
print(Counter(train_time[algos].idxmin(axis=1)))


assert len(test_time[test_time.isna().any(axis=1)]) == 0


# %%
## one hot encoding

from datautils import one_hot_encode_fit, one_hot_encode_transform

y_train_names = list(train_time['algorithm'].to_numpy())
y_test_names = list(test_time['algorithm'].to_numpy())

_, algorithm_to_onehot = one_hot_encode_fit(y_train_names)

y_train_onehot = one_hot_encode_transform(algorithm_to_onehot, y_train_names)
y_test_onehot  = one_hot_encode_transform(algorithm_to_onehot, y_test_names)


# %%

from datautils import string_to_columns

X_train_time, X_columns, _ = string_to_columns(train_time, 'pattern')
X_test_time , X_columns, _ = string_to_columns(test_time, 'pattern')

x_test  = X_test_time[X_columns].to_numpy()
x_train = X_train_time[X_columns].to_numpy()

# %%
# Save the dataset

import pickle
with open('dataset-full.pickle', 'wb') as f:
    pickle.dump([x_train,
    y_train_names,
    x_test,
    y_test_names,
    X_train_time,
    X_test_time,
    train_time,
    test_time,
    y_train_onehot,
    y_test_onehot,
    y_test_time,
    y_train_time,
    X_columns,
    alg_name_to_id,
    algorithms,
    algorithm_to_onehot
    ], f)

