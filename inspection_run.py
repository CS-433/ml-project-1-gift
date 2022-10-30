import numpy as np
import matplotlib.pyplot as plt
from plots import *

#%%########################### FEATURE INSPECTION RUN #########################

# Importing the train dataset and test dataset
train_original, col24_train = load_train_dataset()
test_original, col24_test = load_test_dataset()
names = get_feature_names()

# splitting the dataset into 3 sub-datasets and deleting the constant features
train_datasets = split_jet(train_original, col24_train)
test_datasets = split_jet(test_original, col24_test)
# add the zero-columns for the prediction of the test set
for i in range(len(test_datasets)):
    n_rows = test_datasets[i][0][:,0].size
    col = np.zeros(n_rows)
    test_datasets[i][0] = np.insert(test_datasets[i][0], 1, col, axis=1)

# select the names
labels = []
for i in range(len(train_datasets)):
    labels.append(names[train_datasets[i][2]])

# filling column 3
k_fold = 5
lambdas = np.logspace(-10, 0, 30)

train_filled = []
test_filled = []

for i in range(len(train_datasets)):
    train_jet_filled, test_jet_filled = feature_regression_col3(train_datasets[i][0], test_datasets[i][0], k_fold, lambdas, seed = 1)
    train_filled.append(train_jet_filled)
    test_filled.append(test_jet_filled)

# Visualize features plots and their summary statistics
for i in range(len(train_filled)):
    plot_features_and_stats(train_filled[i][:,2:], i, labels[i])
   
# Visulaize correlation matrixes for each sub dataset
for i in range(len(train_filled)):
    heat_map_correlation_matrix(train_filled[i], i, labels[i])

# removing highly correlated features
train_filled[0] = np.delete(train_filled[0], [4,5,8], 1)
train_filled[1] = np.delete(train_filled[1], [4,5,8,20], 1)
train_filled[2] = np.delete(train_filled[2], [4,8,11,30], 1)

test_filled[0] = np.delete(test_filled[0], [4,5,8], 1)
test_filled[1] = np.delete(test_filled[1], [4,5,8,20], 1)
test_filled[2] = np.delete(test_filled[2], [4,8,11,30], 1)

labels[0] = np.delete(labels[0], [4,5,8])
labels[1] = np.delete(labels[1], [4,5,8,20])
labels[2] = np.delete(labels[2], [4,8,11,30])

# visualizing Empirical Distributions for each feature
for i in range(len(train_filled)):
    yy = train_filled[i][:,1]
    ttx = train_filled[i][:,2:]
    if i == 2:
        jet = [i,i+1]
        print('Jet', jet)
        print('Share of 1:', np.round_(list(yy).count(1)/len(yy), decimals = 2))
        print('Share of -1:',  np.round_(list(yy).count(0)/len(yy), decimals = 2))

    else:
        jet = [i]
        print('Jet', jet)
        print('Share of 1:', np.round_(list(yy).count(1)/len(yy), decimals =2))
        print('Share of -1:',  np.round_(list(yy).count(0)/len(yy), decimals =2))
   
    plot_empirical_distributions(yy, ttx, labels[i], jet)

    print('##################################################################')
    print('##################################################################')
