import sys
sys.path.append("../../ML_project1")

import helpers
from helpers import *
from implementations import * 
# from utilities import *

#%% Importing the train dataset and test dataset

train, col24_train = load_train_dataset()
#%%
test, col24_test = load_test_dataset()

#############################################################################
#%% deleting all the -999 values and their corresponding rows
test_median = subsitute_nan_with_median(test)

train1_del, rows_kept_train1 = delete_nan_rows(train)
test1_del, rows_kept_test1 = delete_nan_rows(test_median)


train1_s, mean1, std1 = standardize_train(train1_del)
test1_s = standardize_test(test1_del, mean1, std1)

cat_cols_train1 = create_dummy(col24_train[rows_kept_train1])
cat_cols_test1 = create_dummy(col24_test[rows_kept_test1])

train1_stack = np.hstack((train1_s, cat_cols_train1))
test1_stack = np.hstack((test1_s, cat_cols_test1))

ids_train1, y_train1, tx_train1 = split_into_ids_y_tx(train1_stack)
ids_test1, y_test1, tx_test1 = split_into_ids_y_tx(test1_stack)

k_fold = 4
lambdas = np.logspace(-10, 0, 30)

lambda1, rmse1 = cross_validation_demo_tx(y_train1, tx_train1, k_fold, lambdas)
w1 = ridge_regression(y_train1, tx_train1, lambda1)[0]

prediction1 = generate_prediction(tx_test1, w1)
print(prediction1)

generate_csv(prediction1, ids_test1, 'sample-submission1.csv')



#############################################################################
#%% deleting all the features with too many -999 values
# and substituting the remaining -999 values with the median of the
# corresponding column

train2_del = delete_feature_with_50pec(train)
test2_del = delete_feature_with_50pec(test)

train2_median = subsitute_nan_with_median(train2_del)
test2_median = subsitute_nan_with_median(test2_del)

train2_s, mean2, std2 = standardize_train(train2_median)
test2_s = standardize_test(test2_median, mean2, std2)

cat_cols_train2 = create_dummy(col24_train)
cat_cols_test2 = create_dummy(col24_test)

train2_stack = np.hstack((train2_s, cat_cols_train2))
test2_stack = np.hstack((test2_s, cat_cols_test2))

ids_train2, y_train2, tx_train2 = split_into_ids_y_tx(train2_stack)
ids_test2, y_test2, tx_test2 = split_into_ids_y_tx(test2_stack)

lambda2, rmse2 = cross_validation_demo_tx(y_train2, tx_train2, k_fold, lambdas)
w2 = ridge_regression(y_train2, tx_train2, lambda1)[0]

prediction2 = generate_prediction(tx_test2, w2)
print(prediction2)

generate_csv(prediction2, ids_test2, 'sample-submission2.csv')



#############################################################################
#%% deleting all the features with too many -999 values
# and substituting the deleting the remaining -999 values 

train3_delf = delete_feature_with_50pec(train)
test3_delf = delete_feature_with_50pec(test)

train3_del, rows_kept_train3 = delete_nan_rows(train3_delf)
test3_median = subsitute_nan_with_median(test3_delf)

train3_s, mean3, std3 = standardize_train(train3_del)
test3_s = standardize_test(test3_median, mean3, std3)

cat_cols_train3 = create_dummy(col24_train[rows_kept_train3])
cat_cols_test3 = create_dummy(col24_test)

train3_stack = np.hstack((train3_s, cat_cols_train3))
test3_stack = np.hstack((test3_s, cat_cols_test3))

ids_train3, y_train3, tx_train3 = split_into_ids_y_tx(train3_stack)
ids_test3, y_test3, tx_test3 = split_into_ids_y_tx(test3_stack)

lambda3 = cross_validation_demo_tx(y_train3, tx_train3, k_fold, lambdas)[0]
w3 = ridge_regression(y_train3, tx_train3, lambda3)[0]

prediction3 = generate_prediction(tx_test3, w3)
print(prediction3)

generate_csv(prediction3, ids_test3, 'sample-submission3.csv')


#############################################################################
#%% polynomial regression to fill the dataset
# with the WHOLE DATASET, except for the categoric column
# this icludes the 6 (or so) columns with too many -999 values

seed = 1
lambdas = lambdas = np.logspace(-10, 0, 30)
degrees = np.array([1, 2, 3, 4])
k_fold = 4

#%%
train4, test4 = feature_regression(train, test, degrees, k_fold, lambdas, seed)







































#%% DEBUG FEATURE_REGRESSION_COL
yy = dataset[:, ind_output[i]].copy()
xx = dataset[:, ind_input[i % n_input]].copy()
#%%
ind_col = np.arange(yy.size)
#%%
y_train, ind_train = delete_nan_elem(yy)
x_train = xx[ind_train]
#%%
ind_train = np.array(ind_train)
ind_train = ind_train.astype(int)

ind_test = []
for i in ind_col:
    if np.count_nonzero(ind_train == i) == 0:
        ind_test.append(i)
ind_test = np.array(ind_test)
ind_test = ind_test.astype(int)
#%%
y_test = yy[ind_test] # -999 values
x_test = xx[ind_test]
#%%
best_degree, best_lambda, best_rmse = best_degree_selection(x_train, y_train, degrees, k_fold, lambdas, seed)
#%%
tx_train = build_poly(x_train, best_degree)
#%%
w = ridge_regression(y_train, tx_train, best_lambda)[0]
#%%
tx_test = build_poly(x_test, best_degree)
y_test = tx_test.dot(w)
#%%    
yy[ind_train] = y_train
yy[ind_test] = y_test









#%% DEBUG FEATURE REGRESSION

trainn = train[:10000,:].copy()
n_rows_train = trainn.shape[0]
testt = test[:10000,:].copy()
n_rows_test = testt.shape[0]
dataset = np.vstack((trainn, testt))
#%%
cnt = count_nan_for_feature(dataset)
#%%
ind = np.arange(cnt.size)

ind_input = []
ind_output = []
for i in ind[2:]:
    if(cnt[i]==0):
        ind_input.append(i)
    else:
        ind_output.append(i)

ind_input = np.array(ind_input)
ind_output = np.array(ind_output)
n_input = len(ind_input)

 #%%       
for i in range(len(ind_output)):
    
    yy = feature_regression_cols(dataset[:, ind_output[i]], dataset[:, ind_input[i % n_input]], degrees, k_fold, lambdas, seed = 1)
                            
    dataset[:, ind_output[i]] = yy
 #%%                           
train_4 = dataset[:n_rows_train, :]
test_4 = dataset[n_rows_train:,:] 

