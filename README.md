# Boosting exact pattern matching with eXtreme Gradient Boosting (and more)

## About

This repository contains datasets and source code for the paper *"Boosting exact pattern matching with eXtreme Gradient Boosting (and more)"*.

## Contents

The experiment consists of two parts.

1. The first part consists of measuring the times of exact pattern matching algorithms to create a dataset for machine learning. 
   - 📁 `smart` - contains the source code for exact pattern matching, which is mostly taken from [SMART tool](https://github.com/smart-tool/smart) (Nov 7 2018, 3b112e362b6ff9e3336b406cc8618164da91ef23).
   - 📁 `data` - contains datasets and patterns excluding datasets that must be downloaded from [Pizza&Chilli](https://pizzachili.dcc.uchile.cl/texts.html) corpus (the tests were performed on 50MB texts).
   - 📄 `run_tests.py` - executes exact pattern matching algorithms and saves the results to `dataset-raw.pickle`
   - 📄 `utils.py` - contains helper functions
   - 📄 `dataset-raw.pickle` - contains raw datasets with pattern searching times for all algorithms

2. The second part is to prepare the dataset for machine learning and train machine learning models whose task is to predict the fastest algorithm for a given pattern.
   - 📄 `start_pred.py` - runs the tests (trainings and predictions)
   - 📄 `datautils.py` - contains helper functions to process the data
   - 📄 `dataset-full.pickle` - processed dataset, prepared for machine learning
   - 📔 `dataset-full-notebook.ipynb` - jupyter notebook with results of exact pattern matching algorithms
   - 📄 `train_datasets.pickle` - train dataset
   - 📄 `test_datasets.pickle` - test dataset
   - 📄 `trained_models.pickle` - contains trained models (make sure to have the same version of scikit-learn to load them, otherwise it will not be possible)
   - 📔 `results-notebook.ipynb` - jupyter notebook with results of machine learning models


**Note:** `*.pickle` files are not included in the repository. They can be downloaded from:
* [datasets](https://tulodz-my.sharepoint.com/:u:/g/personal/robert_susik_p_lodz_pl/ETFKSB8gCIhOkZv1rP7iWy4BfM0DdcSyDdyN0ZO6KTc2ZA?e=mlXKxW)
* [models](https://tulodz-my.sharepoint.com/:u:/g/personal/robert_susik_p_lodz_pl/EXt7LShDGgdMguzSM0yfJJYBRqr634XZIadhR5oYmmuxPw?e=hAtbCS)

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

## Authors

- Robert Susik
- Szymon Grabowski
