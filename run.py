import sys
sys.path.append("../../ml-project-1-gift")

from loading_data import *
from cleaning_dataset import *
from feature_engineering import *
from utilities_linear_regression import *
from utilities_logistic_regression import *
from plots import *
import matplotlib.pyplot as plt
import numpy as np

##############################################################################

#%% LOADING DATASET
### Importing the train dataset and test dataset
train_original, col24_train = load_train_dataset()
test_original, col24_test = load_test_dataset()

#%% CATEGORICAL SPLITTING
### Splitting the dataset into 3 sub-datasets according to the categoric
### feature and deleting the constant features
train_datasets = split_jet(train_original, col24_train)
test_datasets = split_jet(test_original, col24_test)
# add the zero-columns for the prediction of the test set
for i in range(len(test_datasets)):
    n_rows = test_datasets[i][0][:,0].size
    col = np.zeros(n_rows)
    test_datasets[i][0] = np.insert(test_datasets[i][0], 1, col, axis=1)

index_list = [test_datasets[i][1] for i in range(len(test_datasets))]
    
#%% NAN REGRESSION
### Imputing the -999 values of the 3rd column (1st feature) with a ridge 
### regression, using as model matrix all the remaining (complete) features
k_fold = 5
lambdas = np.logspace(-10, 0, 30)

train_filled = []
test_filled = []

for i in range(len(train_datasets)):
    train_jet_filled, test_jet_filled = feature_regression_col3(train_datasets[i][0], test_datasets[i][0], k_fold, lambdas, seed = 1)
    train_filled.append(train_jet_filled)
    test_filled.append(test_jet_filled)
    
#%% HIGLY CORRELATED COLUMNS
### Removing higly correlated columns in each sub-dataset
### The choiche of the columns to remove is the result of the inspection analysis
### (see feature_inspection.py)

train_filled[0] = np.delete(train_filled[0], [4,5,8], 1)
train_filled[1] = np.delete(train_filled[1], [4,5,8,20], 1)
train_filled[2] = np.delete(train_filled[2], [4,8,11,30], 1)

test_filled[0] = np.delete(test_filled[0], [4,5,8], 1)
test_filled[1] = np.delete(test_filled[1], [4,5,8,20], 1)
test_filled[2] = np.delete(test_filled[2], [4,8,11,30], 1)
    

#%% CAP THE OUTLIERS
### 'outlier' : value out of the range mean +- 2*std
### Substituting all the outliers with the corresponding critical value
### Ex: mean = 3, std = 2: 7 --> 5
###                        0 --> 1
train_out = []
test_out = []

for i in range(len(train_filled)):
    tr = fix_outliers(train_filled[i])
    train_out.append(tr)
    te = fix_outliers(test_filled[i])
    test_out.append(te)

#%% CORRECTION OF LONG TAILS FEATURES
### Transforming the right long tailed features
### From data inspection, it turns out they are the non-negative ones
indexes_skew = []
train_skew = []
test_skew = []
for i in range(len(train_out)):
    ind_train = ind_pos_neg(train_out[i])[0]
    ind_train[:2] = False
    tra = correct_skewness(train_out[i], ind_train)
    ind_test = ind_pos_neg(test_out[i])[0]
    ind_test[:2] = False
    tes = correct_skewness(test_out[i], ind_train)
    train_skew.append(tra)
    test_skew.append(tes) 
    
#%% COSINE TRANSFORM OF THE ANGLES
angle_cols0 = np.array([7, 10, 13, 15])
angle_cols1 = np.array([7, 10, 13, 15, 18])
angle_cols23 = np.array([10, 14, 17, 19, 23, 26])
to_correct = [angle_cols0, angle_cols1, angle_cols23]
train_angle = []
test_angle = []

for i in range(len(train_skew)):
    tr = cosine(train_skew[i], to_correct[i])
    te = cosine(test_skew[i], to_correct[i])
    train_angle.append(tr)
    test_angle.append(te)

#%% POLYNOMIAL EXPANSION UP TO THE OPTIMAL DEGREE
### The optimal degree is selected through cross validation over lambdas and degrees
train_exp = []
test_exp = []
best_degrees = []
degrees = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15])
# deg = 15
#%% the following for-loop lasts 10h approximately and chooses the degree 15
### for every sub-dataset
seed = 1
for i in range(len(train_angle)):
    deg = best_degree_dataset(train_angle[i], degrees, k_fold, lambdas, seed)[0]

#%%
for i in range(len(train_angle)):
    curr_exp_train = poly_expansion_blind(train_angle[i], deg)
    train_exp.append(curr_exp_train)
    curr_exp_test = poly_expansion_blind(test_angle[i], deg)
    test_exp.append(curr_exp_test)
    
#%% COUPLED CROSS-PRODUCTS
train_prod = []
test_prod = []

for i in range(len(train_angle)):
    tr = feature_cross_products(train_angle[i])
    te = feature_cross_products(test_angle[i])
    train_prod.append(np.c_[train_exp[i], tr])
    test_prod.append(np.c_[test_exp[i], te])
    
#%% SQUARE ROOT
train_sqrt = []
test_sqrt = []

for i in range(len(train_angle)):
    tr = squareroot(train_angle[i])
    te = squareroot(test_angle[i])
    train_sqrt.append(np.c_[train_prod[i], tr])
    test_sqrt.append(np.c_[test_prod[i], te])

#%% STANDARDIZATION
# notice that the very first feature is the offset, which is a constant column
# of ones, hence it cannot be standardized
train_stand = []
test_stand = []
for i in range(len(train_prod)):
    curr_train, curr_mean, curr_std = standardize_train(train_prod[i])
    curr_test = standardize_test(test_prod[i], curr_mean, curr_std)
    train_stand.append(curr_train)
    test_stand.append(curr_test)    

#%% RIDGE REGRESSION
ws = []
lambdas = np.logspace(-10, 0, 30)
k_fold = 10
lamb = []
train_errors = []
test_errors = []

for i in range(len(train_stand)):
    ids, y, tx = split_into_ids_y_tx(train_stand[i])
    l, best, rmse_tr, rmse_te = cross_validation_demo_tx_lin(y, tx, k_fold, lambdas)
    lamb.append(l)
    w = ridge_regression(y, tx, l)[0]
    ws.append(w)
    train_errors.append(rmse_tr)
    test_errors.append(rmse_te)

#%% PLOT LAMBDAS
plot_lambdas(train_errors, test_errors, lambdas, deg)

#%% COMPUTE CONTINOUS PREDICTIONS
ys = generate_linear_prediction(test_stand, ws)

#%% COMPUTE OPTIMAL THRESHOLD FOR EACH SUB-DATASET
vec_thresholds = np.linspace(-2, 2, 101)
thresholds = []
for i in range(len(train_stand)):
    thr = optimal_threshold(train_stand[i], vec_thresholds, lamb[i])
    thresholds.append(thr)

#%% COMPUTE AND COLLECT BINARY PREDICTIONS ACCORDING TO THE SELECTED THRESHOLD
prediction = collect(ys, index_list, thresholds)

#%% GENERATE THE SUBMISSION
generate_csv(prediction, test_original[:,0], "finalsubmission.csv")

