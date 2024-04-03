import pandas as pd
import numpy as np
import math


def string_to_columns(df:pd.core.frame.DataFrame, column_name:str|int, max_len:int=None, columns:list=None):
    X_columns = []
    if columns is not None:
        X_columns = columns.copy()
    if max_len is None:
        max_len = df[column_name].str.len().max()
    for i in range(max_len):
        col = f'{column_name}_{i}'
        X_columns += [col]
        df[col] = df[column_name].apply(lambda x: ord(x[i]) if len(x) > i else 0)
    return df, X_columns, max_len


def one_hot_encode_fit(data:pd.core.frame.DataFrame|list, column_name:str|int=None):
    if isinstance(data, pd.core.frame.DataFrame):
        if column_name is None:
            raise Exception('column_name cannot be None if data is DataFrame')
        entities = list(data[column_name].unique())
    elif isinstance(data, list):
        entities = list(set(data))
    else:
        raise Exception('Wrong type')
    to_onehot = {}
    for idx, a in enumerate(entities):
        d = np.zeros(len(entities))
        d[idx] = 1
        to_onehot[a] = d
    return entities, to_onehot


def one_hot_encode_transform(to_onehot:dict, data:pd.core.frame.DataFrame|list, column_name:str|int=None):
    if isinstance(data, pd.core.frame.DataFrame):
        if column_name is None:
            raise Exception('column_name cannot be None if data is DataFrame')
        return [to_onehot[el] for el in data[column_name]]
    elif isinstance(data, list):
        return [to_onehot[el] for el in data]
    else:
        raise Exception('Wrong type')


def get_timings(df, index:list=None):
    if index is None:
        index = ['pattern', 'dataset']
    data_transformed = pd.pivot_table(df, 
        values='total_time', 
        index=[el for el in index if el != 'algorithm'],
        columns=['algorithm']
    ).reset_index()
    
    return data_transformed


def comp_lvl_old(p:str, s:str)->float: # log from length of string
    d = {p[:i] : s.count(p[:i]) for i in range(1, len(p)+1)} # liczba wystapien prefiksow w tekscie
    v = list(reversed(d.values()))
    return sum([math.log2((v[i] / v[i-1])) for i in range(1, len(d)) if v[i-1] > 0]) + math.log2((len(s)/v[-1])) # wraz z pierwszym znakiem
    



def H0(s, *args):
  result = 0.0
  len_s = len(s)
  for c in set(s):
    count_c = s.count(c)
    result += count_c / len_s * math.log2(len_s / count_c)
  return result


def H0v2(s, *args):
  result = 0.0
  len_s = len(s)
  for c in set(s):
    count_c = s.count(c)
    result += count_c / len_s * math.log2(count_c / len_s)
  return -result


def H1(s, *args):
  di = {c: [] for c in set(s)}
  for i in range(0, len(s) - 1):
    di[s[i]].append(s[i+1])
  di[s[len(s) - 1]].append("$")  # string terminator
  
  result = 0.0
  for c in set(s):
    result += s.count(c) / len(s) * H0(di[c])
  
  return result


def H2(s, *args):  
  di = {}
  for i in range(0, len(s) - 1):
    key = s[i : i + 2]
    di[key] = []  
  
  for i in range(0, len(s) - 2):
    key = s[i : i+2]
    di[key].append(s[i+2])
  di[s[len(s) - 2 :]].append("$")  # string terminator
  
  result = 0.0
  for key in di:
    result += s.count(key) / len(s) * H0(di[key])
  
  return result




def H0reg(s, *args):
  result = 0.0
  len_s = len(s)
  for c in set(s):
    count_c = s.count(c)
    result += count_c / len_s * math.log2(len_s / count_c)

  # regularization
  
  stats = min(8, math.ceil(math.log2(len_s))) + len(set(s)) * (8 + math.ceil(math.log2(len_s)))
  # min(...) == bits to store the number of alphabet symbols used in s (we assume sigma <= 256)
  # second 8 (bits) to encode a particular symbol
  # math.ceil(math.log2(len_s)) == bits per counters (each value from [1, len_s])
  result += stats

  return result


def H1reg(s, *args):
  di = {c: [] for c in set(s)}
  for i in range(0, len(s) - 1):
    di[s[i]].append(s[i+1])
  di[s[len(s) - 1]].append("$")  # string terminator
  
  result = 0.0
  len_s = len(s)
  for c in set(s):
    result += s.count(c) / len_s * H0(di[c])

  # regularization
  pairs = set()
  for i in range(0, len(s) - 1):
    pairs.add(s[i: i + 2])
  
  min_ = min(8 * 2, math.ceil(math.log2(len_s)))
  # the number of distinct symbol pairs in s may be up to 2^16 (so, 8 * 2 bits are needed), but not more than the length of s
  
  stats = min_ + len(pairs) * (8 * 2 + math.ceil(math.log2(len_s - 1)))
  # 8 * 2 (bits) to encode a particular pair of symbols
  # math.ceil(math.log2(len_s - 1)) == bits per counter (each value from [1, len_s - 1])
  result += stats
  
  return result


def H2reg(s, *args):
  len_s = len(s)
  di = {}
  for i in range(0, len_s - 1):
    key = s[i : i + 2]
    di[key] = []
  
  for i in range(0, len_s - 2):
    key = s[i : i+2]
    di[key].append(s[i+2])
  di[s[len_s - 2 :]].append("$")  # string terminator
  
  result = 0.0
  for key in di:
    result += s.count(key) / len_s * H0(di[key])

  # regularization
  pairs = set()
  for i in range(0, len(s) - 1):
    pairs.add(s[i: i + 2])
  
  min_ = min(8 * 3, math.ceil(math.log2(len_s)))
  # the number of distinct symbol triples in s may be up to 2^24 (so, 8 * 3 bits are needed), but not more than the length of s
  
  stats = min_ + len(pairs) * (8 * 3 + math.ceil(math.log2(len_s - 2)))
  # 8 * 3 (bits) to encode a particular triple of symbols
  # math.ceil(math.log2(len_s - 2)) == bits per counter (each value from [1, len_s - 2])
  result += stats
  
  return result



if __name__ == '__name__':
    v = ['cat', 'ant', 'cat', 'cat', 'bird', 'ant']

    entities, to_onehot = one_hot_encode_fit(v)
    one_hot_encode_transform(to_onehot, v)

    v = pd.DataFrame(v)

    entities, to_onehot = one_hot_encode_fit(v, 0)
    one_hot_encode_transform(to_onehot, v, 0)

    string_to_columns(pd.DataFrame(v), 0)