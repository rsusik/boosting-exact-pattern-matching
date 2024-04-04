# Boosting exact pattern matching with eXtreme Gradient Boosting (and more)

## About

This repository contains datasets and source code for the paper *"Boosting exact pattern matching with eXtreme Gradient Boosting (and more)"*.

## Contents



- 📁 `smart` - contains the source code for exact pattern matching, which is mostly taken from [SMART tool](https://github.com/smart-tool/smart) (Nov 7 2018, 3b112e362b6ff9e3336b406cc8618164da91ef23).
- 📄 `dataset-raw.pickle` - contains raw datasets with pattern searching times for all algorithms.
- 📄 `dataset-full.pickle` - processed dataset, prepared for machine learning
- 📄 `train_datasets.pickle` - train dataset
- 📄 `test_datasets.pickle` - test dataset
- 📄 `trained_models.pickle` - contains trained models (make sure to have the same version of scikit-learn to load them, otherwise it will not be possible)
- 📔 `dataset-full-notebook.ipynb` - jupyter notebook with results of exact pattern matching algorithms
- 📔 `results-notebook.ipynb` - jupyter notebook with results of machine learning models
- 📄 `start_pred.py` - runs the tests (trainings and predictions)
- 📄 `datautils.py` - contains helper functions to process the data

**Note:** `*.pickle` files are not included in the repository. They can be downloaded from:
* [datasets](https://tulodz-my.sharepoint.com/:u:/g/personal/robert_susik_p_lodz_pl/ETFKSB8gCIhOkZv1rP7iWy4BfM0DdcSyDdyN0ZO6KTc2ZA?e=mlXKxW)
* [models](https://tulodz-my.sharepoint.com/:u:/g/personal/robert_susik_p_lodz_pl/EXt7LShDGgdMguzSM0yfJJYBRqr634XZIadhR5oYmmuxPw?e=hAtbCS)



## Requirements

- Python 3.10
- 

## To reproduce the results

1) Clone the repository. `git clone https://github.com/rsusik/boosting-exact-pattern-matching.git`
2) Download the `*.pickle` files from the links above and place them in the root directory of the repository.
3) Run `start_pred.py` script.

*The results may differ in prediction times slightly due to the hardware differences, but the impact on the overall results should be negligible.*

## Authors

- Robert Susik
- Szymon Grabowski
