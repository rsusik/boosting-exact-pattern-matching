#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
from pers import PersistentResults
import inspect
from datautils import comp_lvl_old, H0, H1, H2, H0reg, H1reg, H2reg
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import pickle
import xgboost as xgb
# import catboost as cat
from tqdm import tqdm











def get_supported_kwargs(cls, kwargs:dict)->dict:
    return {k: kwargs[k] for k in kwargs if k in inspect.signature(cls.__init__).parameters.keys()}

def get_rf(**kwargs):
    return RandomForestRegressor(**get_supported_kwargs(RandomForestRegressor, kwargs))

def get_ada(**kwargs):
    return MultiOutputRegressor(AdaBoostRegressor(**get_supported_kwargs(AdaBoostRegressor, kwargs)))

def get_bagg(**kwargs):
    return BaggingRegressor(**get_supported_kwargs(BaggingRegressor, kwargs))

def get_et(**kwargs):
    return ExtraTreesRegressor(**get_supported_kwargs(ExtraTreesRegressor, kwargs))

def get_hgb(**kwargs):
    return MultiOutputRegressor(HistGradientBoostingRegressor(**get_supported_kwargs(HistGradientBoostingRegressor, kwargs)))

def get_xgb(**kwargs):
    return xgb.XGBRegressor(**get_supported_kwargs(xgb.XGBRegressor, kwargs))

# def get_cat(**kwargs):
#     return cat.CatBoostRegressor(loss_function='MultiRMSE', eval_metric='MultiRMSE', **get_supported_kwargs(cat.CatBoostRegressor, kwargs))

def normalize_row(row, n_top):
    row = row.copy()
    top = sorted(row)[:n_top]
    min_top = top[0]
    max_top = top[-1]
    ind = np.where(row > max(top))
    if max_top == min_top:
        row[np.where(row <= max(top))] = 0
    else:
        row = (row - min_top) / (max_top - min_top)
    row[ind] = 1
    return row

def get_y(y_time, n_top):
    return 1-np.array([normalize_row(row, n_top) for row in y_time])




print('Reading raw dataset...')
with open('dataset-full.pickle', 'rb') as f:
    (x_train,       # 00. Pattern (vector of length 256) e.g. array([10, 10, 52, 48, ...
    y_train,        # 01. Class, e.g. 'ebom'
    x_test,         # 02. As x_train but for tests
    y_test,         # 03. As y_train but for tests
    X_train_time,   # 04. DataFrame with all data for training
    X_test_time,    # 05. DataFrame with all data for testing
    Y_train_time,   # 06. Possibly the same as X_train_time
    Y_test_time,    # 07. Possibly the same as X_test_time
    y_train_onehot, # 08. One hot for train
    y_test_onehot,  # 09. One hot for test
    y_test_time,    # 10. Times for each of the tested algorithms (at the time of writing 27)
    y_train_time,   # 11. Times for each of the tested algorithms (at the time of writing 27)
    X_columns,      # 12. Names of columns with subsequent pattern symbols e.g. pattern_0 (first character), etc.
    alg_name_to_id, # 13. Mapping of the algorithm name to its ID
    algorithms,     # 14. Names of all algorithms that have been tested
    algorithm_to_onehot) = pickle.load(f) # 15. Mapping of the algorithm name to its one-hot encoding










if os.path.exists('train_datasets.pickle') and os.path.exists('test_datasets.pickle'):
    print('Loading existing datasets...')
    # load datasets
    with open('train_datasets.pickle', 'rb') as f:
        train_datasets = pickle.load(f)

    with open('test_datasets.pickle', 'rb') as f:
        test_datasets = pickle.load(f)
else:
        
    y_train = np.array(y_train)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # df -> X_train_time
    def get_data(df, y_data_time, x_data, y_data, min_m, max_m, dataset, n_top=3):
        if dataset == 'all':
            data_record_idxs = df[
                (df['m'] >= min_m)&(df['m'] <= max_m)
            ].index
        elif type(dataset) == str:
            data_record_idxs = df[
                (df['m'] >= min_m)&(df['m'] <= max_m)&
                (df['dataset'] == dataset)
            ].index
        elif type(dataset) == list:
            data_record_idxs = df[
                (df['m'] >= min_m)&(df['m'] <= max_m)&
                (df['dataset'].isin(dataset))
            ].index
            dataset = ','.join(sorted(dataset))
        else:
            raise Exception('Wrong dataset type!')
        X_data_time = df.loc[data_record_idxs]
        y_data_time = y_data_time[data_record_idxs,:]
        x_data = x_data[data_record_idxs, :max_m]
        y_data = y_data[data_record_idxs]

        c_data_lvl_old  = np.array([comp_lvl_old(el, el) for el in list(X_data_time['pattern'])]).reshape((-1, 1))
        c_data_lvl_h0   = np.array([H0(el, el) for el in list(X_data_time['pattern'])]).reshape((-1, 1))
        c_data_lvl_h1   = np.array([H1(el, el) for el in list(X_data_time['pattern'])]).reshape((-1, 1))
        c_data_lvl_h2   = np.array([H2(el, el) for el in list(X_data_time['pattern'])]).reshape((-1, 1))
        c_data_lvl_h0reg   = np.array([H0reg(el, el) for el in list(X_data_time['pattern'])]).reshape((-1, 1))
        c_data_lvl_h1reg   = np.array([H1reg(el, el) for el in list(X_data_time['pattern'])]).reshape((-1, 1))
        c_data_lvl_h2reg   = np.array([H2reg(el, el) for el in list(X_data_time['pattern'])]).reshape((-1, 1))
        c_data_mm   = np.array([len(el) for el in list(X_data_time['pattern'])]).reshape((-1, 1))
        c_data_sigm = np.array([len(set(el)) for el in list(X_data_time['pattern'])]).reshape((-1, 1))
        c_data_hist = np.array([[sum(np.array([ord(c) for c in el])==i) for i in range(256)] for el in list(X_data_time['pattern'])])

        y_data_time_norm = get_y(y_data_time, n_top)

        return {
            'min_m': min_m,
            'max_m': max_m,
            'dataset': dataset,
            'n_top': n_top,
            'X_data_time': X_data_time,
            'y_data_time': y_data_time,
            'y_data_time_norm': y_data_time_norm,
            'x_data': x_data,
            'y_data': y_data,
            'c_data_lvl_old': c_data_lvl_old,
            'c_data_lvl_h0': c_data_lvl_h0,
            'c_data_lvl_h1': c_data_lvl_h1,
            'c_data_lvl_h2': c_data_lvl_h2,
            'c_data_lvl_h0reg': c_data_lvl_h0reg,
            'c_data_lvl_h1reg': c_data_lvl_h1reg,
            'c_data_lvl_h2reg': c_data_lvl_h2reg,
            'c_data_mm': c_data_mm,
            'c_data_sigm': c_data_sigm,
            'c_data_hist': c_data_hist,
        }

























    print('Creating training dataset...')

    train_datasets = [] # format: [{'min_m':min_m, 'max_m':max_m, 'dataset':dataset, 'data': dict}]
    for min_m, max_m in tqdm([
        # [4, 4],
        # [6, 6],
        # [8, 8],
        # [16, 16],
        # [32, 32],
        # [64, 64],
        # [128, 128],
        # [256, 256],
        [6, 256],
    ]):
        for dataset in ['all']:
            data = get_data(X_train_time, y_train_time, x_train, y_train, min_m, max_m, dataset)
            train_datasets += [data]

    print('Training dataset created!')


    print('Creating test datasets...')
    test_datasets = [] # format: [{'min_m':min_m, 'max_m':max_m, 'dataset':dataset, 'data': dict}]
    for min_m, max_m in tqdm([
        # [4, 4],
        [6, 6],
        [8, 8],
        [16, 16],
        [32, 32],
        [64, 64],
        [128, 128],
        [256, 256],
        [6, 256],
    ]):
        for dataset in ['data/english/english.50MB', 'data/sources/sources.50MB', 
                        'data/dblp.xml/dblp.xml.50MB', 'data/dna/dna.50MB', 
                        'data/proteins/proteins.50MB', 'data/pitches/pitches.50MB', 'all'
        ]:
            data = get_data(X_test_time, y_test_time, x_test, y_test, min_m, max_m, dataset)
            test_datasets += [data]

    print('Test datasets created!')




    # Save datasets
    print('Saving datasets...')
    with open('train_datasets.pickle', 'wb') as f:
        pickle.dump(train_datasets, f)

    with open('test_datasets.pickle', 'wb') as f:
        pickle.dump(test_datasets, f)


















# Load models if exists
if os.path.exists('trained_models.pickle'):
    print('Loading existing trained models...')
    with open('trained_models.pickle', 'rb') as f:
        trained_models = pickle.load(f)
else:
    print('Starting training...')

    models = [
        # get_cat,
        get_rf,
        get_ada,
        get_bagg,
        get_et,
        get_hgb,
        get_xgb,
    ]

    n_estimators = [
        10#, 100
        # 10
        # 10
        # 1, 5, 25, 50, 75, 200
    ]

    n_tops = [
        # 1, 2, 3, 4
        # 1, 3, 4
        # 1
        3
    ]

    # Model inputs (variants):
    input_columns = [
        ['x_data'],
        ['x_data', 'c_data_mm',  'c_data_sigm', 'c_data_hist'], 
        ['x_data', 'c_data_mm',  'c_data_sigm'], 

        ['x_data', 'c_data_lvl_h0reg', 'c_data_lvl_h1reg', 'c_data_lvl_h2reg', 'c_data_lvl_h0', 'c_data_lvl_h1', 'c_data_lvl_h2', 'c_data_mm',   'c_data_sigm', 'c_data_hist'], 
        ['x_data', 'c_data_lvl_h0reg', 'c_data_lvl_h1reg', 'c_data_lvl_h2reg', 'c_data_lvl_h0', 'c_data_lvl_h1', 'c_data_lvl_h2'],
        ['c_data_lvl_h0reg', 'c_data_lvl_h1reg', 'c_data_lvl_h2reg', 'c_data_lvl_h0', 'c_data_lvl_h1', 'c_data_lvl_h2', 'c_data_mm', 'c_data_sigm'],

        ['x_data', 'c_data_lvl_h0', 'c_data_mm',   'c_data_sigm', 'c_data_hist'], 
        ['x_data', 'c_data_lvl_h0'],
        ['c_data_lvl_h0', 'c_data_mm', 'c_data_sigm'],
        ['x_data', 'c_data_lvl_h1', 'c_data_mm',   'c_data_sigm', 'c_data_hist'], 
        ['x_data', 'c_data_lvl_h1'],
        ['c_data_lvl_h1', 'c_data_mm', 'c_data_sigm'],
        ['x_data', 'c_data_lvl_h2', 'c_data_mm',   'c_data_sigm', 'c_data_hist'], 
        ['x_data', 'c_data_lvl_h2'],
        ['c_data_lvl_h2', 'c_data_mm', 'c_data_sigm'],


        ['x_data', 'c_data_lvl_h0reg', 'c_data_mm',   'c_data_sigm', 'c_data_hist'], 
        ['x_data', 'c_data_lvl_h0reg'],
        ['c_data_lvl_h0reg', 'c_data_mm', 'c_data_sigm'],
        ['x_data', 'c_data_lvl_h1reg', 'c_data_mm',   'c_data_sigm', 'c_data_hist'], 
        ['x_data', 'c_data_lvl_h1reg'],
        ['c_data_lvl_h1reg', 'c_data_mm', 'c_data_sigm'],
        ['x_data', 'c_data_lvl_h2reg', 'c_data_mm',   'c_data_sigm', 'c_data_hist'], 
        ['x_data', 'c_data_lvl_h2reg'],
        ['c_data_lvl_h2reg', 'c_data_mm', 'c_data_sigm'],

    ]





    trained_models = []
    for train_set in (pbar := tqdm(train_datasets)):
        pbar.set_description('Total: ')
        for inputs in (pbar2 := tqdm(input_columns)):
            pbar2.set_description('Inputs: ')
            # get data from train_datasets
            if len(inputs)>1:
                x_train_ex = np.hstack([train_set[el] for el in inputs])
            else:
                x_train_ex = train_set[inputs[0]]
            y_train_time_norm = train_set['y_data_time_norm']
            n_top = train_set['n_top']
            min_m = train_set['min_m']
            max_m = train_set['max_m']
            dataset = train_set['dataset']
            inputs_str = ','.join(sorted(inputs))
            # train models
            for m in models:
                for n_est in n_estimators:
                    model = m(random_state=0, n_estimators=n_est, n_jobs=3)

                    model.fit(x_train_ex, y_train_time_norm)
            
                    trained_models.append({
                        'model': model,
                        'inputs_str': inputs_str,
                        'inputs': inputs,
                        'min_m': min_m,
                        'max_m': max_m,
                        'dataset': dataset,
                        'n_top': n_top,
                        'n_estimators': n_est,
                        'model_name': m.__name__,
                    })


    print('Training done!')


    # Save trained models
    print('Saving trained models...')
    with open('trained_models.pickle', 'wb') as f:
        pickle.dump(trained_models, f)





















print('Starting testing...')
input('PRESS ENTER TO CONTINUE.')

results = PersistentResults(
    'results.pickle', # old models.results.norm.pickle
    interval=1,
    tmpfilename=f'~results.pickle.tmp',
    result_prefix='',
    skip_list=['model', 'x_test_ex', 'y_test_time', 'y_test', 'algorithms', 'data']
)

def get_result(model, dataset, min_m, max_m, x_test_ex, y_test_time, y_test, algorithms, *args, **kwargs):
    # Prediction for all test data
    t_full_time_list = []
    for i in range(11):
        t_start = time.time()
        pred = model.predict(x_test_ex)
        t_end = time.time()
        t_full_time_list.append(t_end - t_start)
    t_full_time_avg = sum(t_full_time_list) / len(t_full_time_list) # srednia
    t_full_time_median = sorted(t_full_time_list)[len(t_full_time_list)//2] # mediana
    # The same for single prediction
    t_single_time = []
    for i in range(101):
        t_start = time.time()
        _ = model.predict([x_test_ex[i]])
        t_end = time.time()
        t_single_time.append(t_end - t_start)
    t_single_time_avg = sum(t_single_time) / len(t_single_time)
    t_single_time_median = sorted(t_single_time)[len(t_single_time)//2]
    indices = np.argmax(pred, axis=1)
    acc = sum(np.array([algorithms[el] == y_test[idx] for idx, el in enumerate(indices)])) / len(indices)
    perfect_result = np.array([min(el) for el in y_test_time])
    ebom_idx = algorithms.index('ebom')
    ebom_result = np.array([el[ebom_idx] for el in y_test_time])
    this_result = np.array([y_test_time[idx][el] for idx, el in enumerate(indices)])
    response_alg = [algorithms[el] for el in indices]
    ratio_norm = sum(this_result / perfect_result) / len(indices)
    # Results according to the formula \sum(X / X_{perfect}) / N, where X is the time of the algorithm and X_{perfect} is the ideal time
    # Similar with the prediction time \sum( (X+t_{pred}) / X_{perfect} ) / N
    current_ratio_norm_with_pred_full_time                  = sum((this_result + ((t_full_time_avg)/len(indices))) / perfect_result)/len(indices)
    current_ratio_norm_with_pred_single_time_avg            = sum((this_result + t_single_time_avg) / perfect_result)/len(indices)
    current_ratio_norm_with_pred_single_time_median         = sum((this_result + t_single_time_median) / perfect_result)/len(indices)
    current_ratio_norm_to_ebom_with_pred_full_time          = sum((this_result + ((t_full_time_avg)/len(indices))) / ebom_result)/len(indices)
    current_ratio_norm_to_ebom_with_pred_single_time_avg    = sum((this_result + t_single_time_avg) / ebom_result)/len(indices)
    current_ratio_norm_to_ebom_with_pred_single_time_median = sum((this_result + t_single_time_median) / ebom_result)/len(indices)
    perfect_ratio_norm_with_pred_full_time                  = sum((perfect_result + ((t_full_time_avg)/len(indices))) / perfect_result)/len(indices)
    perfect_ratio_norm_with_pred_single_time_avg            = sum((perfect_result + t_single_time_avg) / perfect_result)/len(indices)
    perfect_ratio_norm_with_pred_single_time_median         = sum((perfect_result + t_single_time_median) / perfect_result)/len(indices)
    # Below results for the sum of times, i.e. \sum(X) / \sum(X_{perfect})
    perfect = sum(perfect_result)
    current = sum(this_result)
    ratio = current / perfect
    current_with_pred_full_time          = current + t_full_time_avg
    current_with_pred_single_time_avg    = current + t_single_time_avg * len(indices)
    current_with_pred_single_time_median = current + t_single_time_median * len(indices)
    perfect_with_pred_full_time          = perfect + t_full_time_avg
    perfect_with_pred_single_time_avg    = perfect + t_single_time_avg * len(indices)
    perfect_with_pred_single_time_median = perfect + t_single_time_median * len(indices)
    ratio_with_pred_full_time = current_with_pred_full_time / perfect_with_pred_full_time
    ratio_with_pred_single_time_avg = current_with_pred_single_time_avg / perfect_with_pred_single_time_avg
    ratio_with_pred_single_time_median = current_with_pred_single_time_median / perfect_with_pred_single_time_median
    
    return {
        'dataset': dataset,
        'min_m': min_m,
        'max_m': max_m,
        'acc': acc,
        'ratio': ratio,
        'full_pred_time': t_full_time_avg,
        'full_pred_time_median': t_full_time_median,
        'single_pred_time': t_single_time,
        'perfect': perfect,
        'current': current,
        'this_result': this_result,
        'perfect_result': perfect_result,
        'ratio_norm': ratio_norm,
        'current_ratio_norm_with_pred_full_time': current_ratio_norm_with_pred_full_time,
        'current_ratio_norm_with_pred_single_time_avg': current_ratio_norm_with_pred_single_time_avg,
        'current_ratio_norm_with_pred_single_time_median': current_ratio_norm_with_pred_single_time_median,
        'current_ratio_norm_to_ebom_with_pred_full_time': current_ratio_norm_to_ebom_with_pred_full_time,
        'current_ratio_norm_to_ebom_with_pred_single_time_avg': current_ratio_norm_to_ebom_with_pred_single_time_avg,
        'current_ratio_norm_to_ebom_with_pred_single_time_median': current_ratio_norm_to_ebom_with_pred_single_time_median,
        'perfect_ratio_norm_with_pred_full_time': perfect_ratio_norm_with_pred_full_time,
        'perfect_ratio_norm_with_pred_single_time_avg': perfect_ratio_norm_with_pred_single_time_avg,
        'perfect_ratio_norm_with_pred_single_time_median': perfect_ratio_norm_with_pred_single_time_median,
        'current_with_pred_full_time': current_with_pred_full_time,
        'current_with_pred_single_time_avg': current_with_pred_single_time_avg,
        'current_with_pred_single_time_median': current_with_pred_single_time_median,
        'perfect_with_pred_full_time': perfect_with_pred_full_time,
        'perfect_with_pred_single_time_avg': perfect_with_pred_single_time_avg,
        'perfect_with_pred_single_time_median': perfect_with_pred_single_time_median,
        'ratio_with_pred_full_time': ratio_with_pred_full_time,
        'ratio_with_pred_single_time_avg': ratio_with_pred_single_time_avg,
        'ratio_with_pred_single_time_median': ratio_with_pred_single_time_median,
        'num_of_patterns': len(indices),
    }


for test_set in (pbar := tqdm(test_datasets)):
    pbar.set_description('Total: ')
    for trained_model in (pbar2 := tqdm(trained_models)):
        pbar2.set_description('Models: ')
        inputs = trained_model['inputs']
        inputs_str = ','.join(sorted(inputs))
        n_top = trained_model['n_top']
        n_est = trained_model['n_estimators']

        if len(inputs)>1:
            x_test_ex = np.hstack([test_set[el] for el in inputs])
        else:
            x_test_ex = test_set[inputs[0]]
        y_test_time_norm = test_set['y_data_time_norm']
        y_test_time = test_set['y_data_time']
        y_test = test_set['y_data']
        min_m = test_set['min_m']
        max_m = test_set['max_m']
        dataset = test_set['dataset']

        train_dataset = trained_model['dataset']
        model_min_m = trained_model['min_m']
        model_max_m = trained_model['max_m']

        # and not (...) because we want to test on models trained on full range
        if (trained_model['min_m'] != min_m or trained_model['max_m'] != max_m) and \
            not (trained_model['min_m'] == 6 and trained_model['max_m'] == 256): # 
            continue

        # != 'all' because we want to test on models trained on all datasets
        if train_dataset != dataset and train_dataset != 'all':
            continue

        # If the model is trained on a larger pattern, we complete the pattern with zeros and keep the rest of the parameters
        # Warning: assumption that the pattern (x_data) is always at the beginning
        if 'x_data' in inputs and trained_model['max_m'] > max_m:
            x_test_ex = np.hstack((x_test_ex[:, :max_m], np.zeros((len(x_test_ex), trained_model['max_m'] - max_m)), x_test_ex[:, max_m:]))
        # TODO: it is possible to add a case when the model is trained on a smaller pattern, then the pattern should be cut off
        # if 'x_data' not in inputs:
        #     min_m = 0
        #     max_m = 0

        model = trained_model['model']

        # if results.any( inputs_str=[inputs_str], min_m=[min_m], max_m=[max_m], n_top=[n_top],  n_estimators=[n_est],  model_name=trained_model['model_name'], model_min_m=[model_min_m], model_max_m=[model_max_m],):
        if len([x for x in results.data if all([
            x['dataset']==dataset,
            x['train_dataset']==train_dataset,
            x['inputs_str']==inputs_str,
            x['min_m']==min_m,
            x['max_m']==max_m,
            x['n_top']==n_top,
            x['n_estimators']==n_est, 
            x['model_name']  ==trained_model['model_name'],
            x['model_min_m'] ==model_min_m,
            x['model_max_m'] ==model_max_m
        ])]) > 0:
            print('skipping -> ', str({'n_top':n_top, 'n_estimators':n_est, 'model_name':trained_model['model_name'],'inputs_str':inputs_str,'min_m':min_m,'max_m':max_m,'model_min_m':model_min_m, 'model_max_m':model_max_m,}))
            continue

        # warm up
        pred = model.predict(x_test_ex)
        indices = np.argmax(pred, axis=1)

        results.append(get_result, 
            model, dataset, min_m, max_m, x_test_ex, y_test_time, y_test, algorithms,
            n_top=n_top, n_estimators=n_est, model_name=trained_model['model_name'],
            inputs_str=inputs_str, 
            model_min_m=model_min_m, model_max_m=model_max_m,
            train_dataset=train_dataset,
        )





        