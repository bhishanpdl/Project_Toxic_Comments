#!/usr/bin/env python

__doc__ = """
Author: Bhishan Poudel

We must start a project with train test split.
We do all the modelling on train data and then test the
model performance on test data in the end.


"""

# Imports
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# local imports
import config

# random state
SEED = 100 # keep it fixed here, do not use from config file
np.random.seed(SEED)

# params
path_data_raw    = config.path_data_raw
path_data_train  = config.path_data_train
path_data_test   = config.path_data_test
path_data_sample = config.path_data_sample
train_size       = config.train_size
compression      = config.compression

# Load the data
df = pd.read_csv(path_data_raw,compression=compression)

# train test split
df_train, df_test = train_test_split(
    df,
    train_size=train_size,
    random_state=SEED
    )

# prints
print(f"df       : {df.shape}")
print(f"df_train : {df_train.shape}")
print(f"df_test  : {df_test.shape}")

# write files
# NOTE: If we use compression='zip' in file writing, unzipping file gives
#       binary file, so first create unzipped files, then zip it.
# NOTE: We must use 'zip' command to zip file
#       if we use macos finder and zip it, pandas can not read it.
f1 = path_data_train.rstrip('.zip')
f2 = path_data_test.rstrip('.zip')
df_train.head().to_csv(path_data_sample,index=False)
df_train.to_csv(f1,index=False)
df_test.to_csv(f2,index=False)

# compress the data on your local machine
# zip train.csv.zip train.csv