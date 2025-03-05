# Boosting exact pattern matching with eXtreme Gradient Boosting (and more)

## About

This repository contains datasets and source code for the paper *"Boosting exact pattern matching with eXtreme Gradient Boosting (and more)"*.

## Dataset for ML

For those who want to perform the research on our dataset, we provide it in convenient json format in [dataset_ml_json](dataset_ml_json) folder.

To reproduce our results, please follow the instructions below.

## Contents

The experiment consists of two parts.

1. The first part consists of measuring the times of exact pattern matching algorithms to create a dataset for machine learning. 
   - 📁 `smart` - contains the source code for exact pattern matching, which is mostly taken from [SMART tool](https://github.com/smart-tool/smart) (Nov 7 2018, 3b112e362b6ff9e3336b406cc8618164da91ef23).
   - 📁 `data` - contains datasets and patterns excluding datasets that must be downloaded from [Pizza&Chilli](https://pizzachili.dcc.uchile.cl/texts.html) corpus (the tests were performed on 50MB texts).
   - 📄 `run_tests.py` - executes exact pattern matching algorithms and saves the results to `dataset-raw.pickle`
   - 📄 `utils.py` - contains helper functions
   - 📄 `dataset-raw.pickle` - contains raw datasets with pattern searching times for all algorithms

2. The second part is to prepare the dataset for machine learning and train models to predict the fastest algorithm for a given pattern.
   - 📄 `start_pred.py` - runs the tests (trainings and predictions)
   - 📄 `datautils.py` - contains helper functions
   - 📄 `dataset-full.pickle` - processed dataset, prepared for machine learning
   - 📔 `dataset-full-notebook.ipynb` - jupyter notebook with results of exact pattern matching algorithms
   - 📄 `train_datasets.pickle` - train dataset
   - 📄 `test_datasets.pickle` - test dataset
   - 📄 `trained_models.pickle` - contains trained models (make sure to have the same version of scikit-learn to load them, otherwise it will not be possible)
   - 📔 `results-notebook.ipynb` - jupyter notebook with results of machine learning models


**Note:** `*.pickle` files are not included in the repository. They can be downloaded from:
* [datasets](https://tulodz-my.sharepoint.com/:u:/g/personal/robert_susik_p_lodz_pl/ESWnEWdN0MFPldZgqmjljHwBlbVuylqQzHsf91YFZ2fPuw?e=L2Ovh3)
* [models](https://tulodz-my.sharepoint.com/:u:/g/personal/robert_susik_p_lodz_pl/ESAuixQRzZBKslk3w4ZshmkBioxYgLwxwhO44W3XQcGIsw?e=hhdzUN)

**Note 2:** The text corpus is also not included in the repository. It can be downloaded from [Pizza&Chilli](https://pizzachili.dcc.uchile.cl/texts.html) (the tests were performed on 50MB texts) and then placed in the `data` directory.

## Requirements

- Python 3.10.10 *(other versions may work, but they were not tested)*
- All required packages are listed in `requirements.txt` file.

## To reproduce the results

1) Clone the repository 
   
   ```bash
   git clone https://github.com/rsusik/boosting-exact-pattern-matching.git
   ```
   
2) Install required packages by running 
   
   ```bash
   pip install -r requirements.txt
   ```

3) Download the `*.pickle` files from the links above and place them in the root directory of the repository.

4) Run `start_pred.py` script.

*The results may differ in prediction times slightly due to the hardware differences, but the impact on the overall results should be negligible.*

## To create the dataset from scratch

Perform step 1 and 2 from the previous section and then:

3) Download the text corpus from [Pizza&Chilli](https://pizzachili.dcc.uchile.cl/texts.html) (the tests were performed on 50MB texts) and place it in the `data` directory.

4) Execute searching algorithms to create the dataset:
   
   ```bash
   cd smart
   python compile.py
   cd ..
   python run_tests.py
   ```

*Running the searching algorithms may take a long time and lead to slightly different results due to the hardware differences.*


## List of algorithms used in the experiment

| # | Algorithm | Source |
| --- | --- | --- |
| 1 | ac | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/ac.c) |
| 2 | ag | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/ag.c) |
| 3 | akc | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/akc.c) |
| 4 | askip | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/askip.c) |
| 5 | aut | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/aut.c) |
| 6 | bf | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/bf.c) |
| 7 | bfs | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/bfs.c) |
| 8 | blim | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/blim.c) |
| 9 | bm | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/bm.c) |
| 10 | bmh-sbndm | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/bmh-sbndm.c) |
| 11 | bndm | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/bndm.c) |
| 12 | bndml | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/bndml.c) |
| 13 | bom | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/bom.c) |
| 14 | br | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/br.c) |
| 15 | bsdm | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/bsdm.c) |
| 16 | bww | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/bww.c) |
| 17 | bxs | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/bxs.c) |
| 18 | dbww | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/dbww.c) |
| 19 | ebom | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/ebom.c) |
| 20 | epsm | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/epsm.c) |
| 21 | fbom | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/fbom.c) |
| 22 | fdm | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/fdm.c) |
| 23 | ffs | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/ffs.c) |
| 24 | fjs | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/fjs.c) |
| 25 | fndm | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/fndm.c) |
| 26 | fs | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/fs.c) |
| 27 | fsbndm | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/fsbndm.c) |
| 28 | graspm | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/graspm.c) |
| 29 | hor | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/hor.c) |
| 30 | iom | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/iom.c) |
| 31 | jom | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/jom.c) |
| 32 | kbndm | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/kbndm.c) |
| 33 | kmp | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/kmp.c) |
| 34 | kmpskip | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/kmpskip.c) |
| 35 | kr | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/kr.c) |
| 36 | ksa | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/ksa.c) |
| 37 | lbndm | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/lbndm.c) |
| 38 | ldm | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/ldm.c) |
| 39 | mp | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/mp.c) |
| 40 | ms | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/ms.c) |
| 41 | nsn | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/nsn.c) |
| 42 | om | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/om.c) |
| 43 | pbmh | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/pbmh.c) |
| 44 | qlqs | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/qlqs.c) |
| 45 | qs | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/qs.c) |
| 46 | raita | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/raita.c) |
| 47 | rcolussi | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/rcolussi.c) |
| 48 | rf | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/rf.c) |
| 49 | sa | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/sa.c) |
| 50 | sabp | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/sabp.c) |
| 51 | sbndm-bmh | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/sbndm-bmh.c) |
| 52 | sbndm | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/sbndm.c) |
| 53 | sebom | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/sebom.c) |
| 54 | sfbom | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/sfbom.c) |
| 55 | simon | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/simon.c) |
| 56 | skip | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/skip.c) |
| 57 | smith | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/smith.c) |
| 58 | smoa | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/smoa.c) |
| 59 | so | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/so.c) |
| 60 | ssabs | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/ssabs.c) |
| 61 | ssef | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/ssef.c) |
| 62 | ssm | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/ssm.c) |
| 63 | tbm | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/tbm.c) |
| 64 | tndm | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/tndm.c) |
| 65 | tndma | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/tndma.c) |
| 66 | trf | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/trf.c) |
| 67 | ts | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/ts.c) |
| 68 | tsa | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/tsa.c) |
| 69 | tsw | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/tsw.c) |
| 70 | tunedbm | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/tunedbm.c) |
| 71 | tvsbs | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/tvsbs.c) |
| 72 | tw | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/tw.c) |
| 73 | twfr | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/twfr.c) |
| 74 | wc | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/wc.c) |
| 75 | wfr | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/wfr.c) |
| 76 | wom | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/wom.c) |
| 77 | ww | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/ww.c) |
| 78 | zt | [source](https://github.com/rsusik/boosting-exact-pattern-matching/tree/main/smart/source/algos/zt.c) |



## Authors

- Robert Susik
- Szymon Grabowski
