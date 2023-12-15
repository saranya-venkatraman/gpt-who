# Implementation of "GPT-who: An Information Density-based Machine-Generated Text Detector"


This repository provides code to calculate the 4 UID-based features and UID minimum and maximum span features described in the paper for efficient and accurate machine text detection.

## Installation

Use the package manager [pip](https://pypi.org/project/pip/) to install all requirements.

```bash
$ pip install -r requirements.txt
```

## This repository contains 2 scripts:
1. [get_uid_features.py](https://github.com/saranya-ven/gpt-who/blob/main/get_uid_features.py): This scripts loads texts and author labels from a csv file/any data source, calculates all UID features needed for GPT-who and writes them to a new csv file. This new generated csv file is the input to [gpt-who.py](https://github.com/saranya-ven/gpt-who/blob/main/gpt-who.py) 

#### Arguments
```--input_path: Path to the CSV file or data source containing text and corresponding labels (default: None).
--cache_path: Path to the cache directory for the GPT-2 XL model (default: "./.cache/models/gpt2-xl").
--output_path: Path to the CSV file where UID features will be saved (default: "./scores/uid_features.csv").
```

#### Example Usage 
```
python gptwho_uid_features.py --input_path ./data/text_labels.csv --cache_path ./model_cache/gpt2-xl --output_path ./scores/uid_features.csv
```

2. [gpt-who.py](https://github.com/saranya-ven/gpt-who/blob/main/gpt-who.py): This script takes as input two .csv files with UID features corresponding to the train and test split of the dataset, calculates the UID span features to concatenate with the other 4 (uid_var, uid_diff, uid_diff2, and mean), runs logistic regression, predicts labels, and reports machine text detection performance.

#### Arguments
```
--train_file: Path to the CSV file containing UID features for the training split (default: "./scores/train_uid_scores.csv").
--test_file: Path to the CSV file containing UID features for the test split (default: "./scores/test_uid_scores.csv").
```
#### Example Usage

```
python uid_span_features_logreg.py --train_file ./data/train_uid_features.csv --test_file ./data/test_uid_features.csv
```

[./scores folder](https://github.com/saranya-ven/gpt-who/tree/main/scores): We also provide UID feature train and test files for the [ArguGPT](https://arxiv.org/pdf/2304.07666.pdf) dataset as an example dataset to run this code. However, our method can be applied to any custom dataset with "text" and "label" fields corresponding to the textual content and author labels.
