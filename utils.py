import os
import string
import pickle
import shutil
import subprocess
from tqdm import tqdm
from typing import List
from pers import PersistentResults

def run_test(algfilename, pattern, dataset, n, exceptionOk=True):
    try:
        cmd = [f'./smart/source/bin/{algfilename}', f'{pattern}', f'{len(pattern)}', dataset, str(n)]
        out = subprocess.check_output(
            cmd
        ).decode('utf-8')
        
        try:
            p, m, s, nn, pt, rt, occ = out.replace('\n', '').split('\t')
        except Exception as ex:
            print(' '.join(cmd))
            print(out)
            raise ex
        return {
            'dataset': dataset,
            'pattern': pattern,
            'm': len(pattern),
            't': s,
            'n': int(n),
            'nn': int(nn),
            'algorithm': algfilename,
            'pre_time': float(pt),
            'run_time': float(rt),
            'total_time': float(rt) + float(pt),
            'occurences': int(occ),
        }
    except subprocess.CalledProcessError as ex:
        return {
            'dataset': dataset,
            'pattern': pattern,
            'm': len(pattern),
            't': None,
            'n': int(n),
            'nn': None,
            'algorithm': algfilename,
            'pre_time': None,
            'run_time': None,
            'total_time': None,
            'occurences': None,
            'exception': ex,
            'isException': True
        }



def get_tmp_dataset_name(dataset_path, dataset):
    if dataset_path is None:
        return dataset # it means the dataset contains path to dataset
    else:
        return f'{dataset_path}/{os.path.basename(dataset)}'

def run_tests(r=100, algos:List[str]=None, datasets:List[str]=None, dataset_path:str=None, copy_datasets=False)->PersistentResults:
    if copy_datasets: # copy datasts to RAM (i.e. /tmp folder)
        os.makedirs(dataset_path, exist_ok=True)
        for dataset in datasets:
            if not os.path.isfile(get_tmp_dataset_name(dataset_path, dataset)):
                shutil.copyfile(dataset, get_tmp_dataset_name(dataset_path, dataset))


    if algos is None:
        algos = []
        for filename in os.listdir('smart/source/bin/'):
            if len(filename.split('.'))>1: continue # pomija źródłowe
            if len(set(string.digits).intersection(set(filename))) > 0: continue # pomija warianty
            algos.append(filename)

    results = PersistentResults(
        f'results-r{r}.pickle',    # do jakiego pliku zapisywac
        interval=r,    # co x wzorców
        tmpfilename=f'~results-r{r}.pickle',
        result_prefix=''
    )

    for dataset in datasets:
        print(f'Testing {dataset}')
        patterns = pickle.load(open(f'{dataset}-r{r}.patterns', 'rb'))
        # patterns.append('ABCDE1234')
        dataset = get_tmp_dataset_name(dataset_path, dataset)
        s = open(f'{dataset}', 'rb').read()
        n = len(s)

        if results.all(                
            algfilename=algos, 
            pattern=patterns, 
            dataset=[dataset], 
            n=[n], 
            exceptionOk=[False]
        ):
            continue

        pbar = tqdm(algos)
        for filename in pbar:
            if results.all(                
                algfilename=[filename], 
                pattern=patterns, 
                dataset=[dataset], 
                n=[n], 
                exceptionOk=[False]
            ):
                continue

            pbar.set_description(f'alg:{filename}')
            # pbar.set_description(f'alg:{filename}|{sum(times)/max(1, len(times)):.4f}s pp')
            # INFO: Warmup
            run_test(
                algfilename=filename, 
                pattern='ABCDEFGH', 
                dataset=dataset, 
                n=n, 
                exceptionOk=False
            )
            for pattern in patterns:
                if len(pattern)<4: continue;
                results.append(run_test, 
                    algfilename=filename, 
                    pattern=pattern, 
                    dataset=dataset, 
                    n=n, 
                    exceptionOk=False
                )
        results.save()
    return results