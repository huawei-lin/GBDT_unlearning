# OnlineBoosting

The implementation for paper [Machine Unlearning in Gradient Boosting Decision Trees](https://openreview.net/forum?id=1ciFPLlyR6d) (Accepted on SIGKDD 2023).
OnlineBoosting support training, unlearning and tuning. This implementation base on the toolkit of [ABCBoost](https://github.com/pltrees/abcboost).

## Quick Start
### Installation guide
Run the following commands to build ABCBoost from source:
```
git clone https://github.com/huawei-lin/OnlineBoosting.git
cd OnlineBoosting
mkdir build
cd build
cmake ..
make
cd ..
```
This will create three executables (`abcboost_train`, `abcboost_predict`, `abcboost_unlearn`, `abcboost_tune`, and `abcboost_clean`) in the `abcboost` directory.
`abcboost_train` is the executable to train models.
`abcboost_predict` is the executable to validate and inference using trained models.
`abcboost_unlearn` is the executable to unlearn a given collection of training data from a trained model.
`abcboost_tune` is the executable to tune a trained model in new data.
`abcboost_clean` is the executable to clean csv data.

The default setting builds ABCBoost as a single-thread program.  To build ABCBoost with multi-thread support [OpenMP](https://en.wikipedia.org/wiki/OpenMP) (OpenMP comes with the system GCC toolchains on Linux), turn on the multi-thread option:
```
cmake -DOMP=ON ..
make clean
make
```
Note that the default g++ on Mac may not support OpenMP.  To install, execute `brew install libomp` before `cmake`.


If we set `-DNATIVE=ON`, the compiler may better optimize the code according to specific native CPU instructions: 
```
cmake -DOMP=ON -DNATIVE=ON .. 
make clean
make
```
We do not recommend to turn on this option on Mac. 

### Datasets 

Two datasets are provided under `data/` folder: [pendigits](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits) and [optdigits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).


## More Configuration Options:
#### Data related:
* `-data_min_bin_size` minimum size of the bin
* `-data_max_n_bins` max number of bins (default 1000)
* `-data_path, -data` path to train/test data
#### Tree related:
* `-tree_max_n_leaves`, -J (default 20)
* `-tree_min_node_size` (default 10)
* `-tree_n_random_layers` (default 0)
* `-feature_split_sample_rate` (default 1.0)
#### Model related:
* `-model_data_sample_rate` (default 1.0)
* `-model_feature_sample_rate` (default 1.0)
* `-model_shrinkage`, `-shrinkage`, `-v`, the learning rate (default 0.1)
* `-model_n_iterations`, `-iter` (default 1000)
* `-model_n_classes` (default 0) the max number of classes allowed in this model (>= the number of classes on current dataset, 0 indicates do not set a specific class number)
* `-model_name`, `-method` regression/lambdarank/mart/abcmart/robustlogit/abcrobustlogit (default robustlogit)
#### Unlearning related:
* `-unlearning_ids_path` path to unlearning indices
* `-lazy_update_freq` (default 1)
#### Tuning related:
* `-tuning_data_path` path to tuning data
#### Parallelism:
* `-n_threads`, `-threads` (default 1)
#### Other:
* `-save_log`, 0/1 (default 0) whether save the runtime log to file
* `-save_model`, 0/1 (default 1)
* `-save_prob`, 0/1 (default 0) whether save the prediction probability for classification tasks
* `-save_importance`, 0/1 (default 0) whether save the feature importance in the training
