import os

# data
dat_dir = os.path.join('..','data')

path_data_raw = os.path.join(dat_dir, 'raw', 'jigsaw_toxic.csv.zip')
path_data_train = os.path.join(dat_dir, 'raw', 'train.csv.zip')
path_data_test = os.path.join(dat_dir, 'raw', 'test.csv.zip')
path_data_sample = os.path.join(dat_dir, 'raw', 'sample.csv')
compression = 'zip'

# params
train_size = 0.8
test_size = 1-train_size
SEED = 100

model_type = 'regression'
target = None # this is multilabel classification