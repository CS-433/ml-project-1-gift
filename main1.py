import sys
sys.path.append("../../ML_project1")

from helpers import *
from implementations import * 
from utilities import *
# from feature_engineering import *
# from utilities import *
# from tqdm import tqdm

#%% Importing the train dataset and test dataset
train_original, col24_train = load_train_dataset()
test_original, col24_test = load_test_dataset()

#%% splitting the dataset into 3 sub-datasets
# and deleting the constant features
train_datasets = split_jet(train_original, col24_train)
test_datasets = split_jet(test_original, col24_test)
for i in range(len(test_datasets)):
    n_rows = test_datasets[i][0][:,0].size
    col = np.zeros(n_rows)
    test_datasets[i][0] = np.insert(test_datasets[i][0], 1, col, axis=1)
    
#%% deleting the features with more than 50% of nan values
ttrain = []
ttest = []
for i in range(len(train_datasets)):
    ttrain.append(delete_feature_with_50pec(train_datasets[i][0]))
    ttest.append(delete_feature_with_50pec(test_datasets[i][0]))

##############################################################################    
#%% impute the missing values with the median
for i in range(len(ttrain)):
    ttrain[i] = subsitute_nan_with_median(ttrain[i])
    ttest[i] = subsitute_nan_with_median(ttest[i])
##############################################################################
    
#%% LINEAR polynomial expansion of the dataset
expanded_train = []
expanded_test = []
deg_sel_train = []
degrees = np.arange(1,8)
k_fold = 4
lambdas = np.logspace(-10, 0, 30)
for i in range(len(ttrain)):
    curr_expansion, deg_sel = poly_expansion_lin(ttrain[i], degrees, k_fold, lambdas, seed = 1)
    expanded_train.append(curr_expansion)
    deg_sel_train.append(deg_sel)
    
