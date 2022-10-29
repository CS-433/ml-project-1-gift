from utilities_linear_regression import *
from feature_engineering import *
# from helpers import *
# from implementations import * 
# from utilities import *
# from feature_engineering import *
# from utilities import *
import numpy as np
import matplotlib.pyplot as plt    

#%% Importing the train dataset and test dataset
train_original, col24_train = load_train_dataset()
test_original, col24_test = load_test_dataset()

#%% splitting the dataset into 3 sub-datasets and deleting the constant features
train_datasets = split_jet(train_original, col24_train)
test_datasets = split_jet(test_original, col24_test)
# add the zero-columns for the prediction of the test set
for i in range(len(test_datasets)):
    n_rows = test_datasets[i][0][:,0].size
    col = np.zeros(n_rows)
    test_datasets[i][0] = np.insert(test_datasets[i][0], 1, col, axis=1)
    
#%% filling column 3
k_fold = 5
lambdas = np.logspace(-10, 0, 30)

train_filled = []
test_filled = []

for i in range(len(train_datasets)):
    train_jet_filled, test_jet_filled = feature_regression_col3(train_datasets[i][0], test_datasets[i][0], k_fold, lambdas, seed = 1)
    train_filled.append(train_jet_filled)
    test_filled.append(test_jet_filled)
   


#%% Visualize features plots and their summary statistics

for i in range(3):
    Plot_Features_and_stats(train_filled[i][:,2:],i)


   
#%% Visulaize correlation matrixes for each sub dataset

for i in range(len(train_filled)):
    heat_map_corrmatr(train_filled[i],i)


#%% Visualize Empirical Distributions for each feature

for i in range(len(train_filled)):
        
    yy = train_filled[i][:,1]
    ttx = train_filled[i][:,2:]
    if i == 2:
        print('Jet',[i,i+1])
        print('Share of 1:', np.round_(list(yy).count(1)/len(yy), decimals =2))
        print('Share of -1:',  np.round_(list(yy).count(0)/len(yy), decimals =2))

    else:
        print('Jet',i)
        print('Share of 1:', np.round_(list(yy).count(1)/len(yy), decimals =2))
        print('Share of -1:',  np.round_(list(yy).count(0)/len(yy), decimals =2))
    Plot_empirical_distributions(yy, ttx, train_filled[i])

    print('###########################################')
    print('###########################################')
    
    
#%%


    

#%%



#%% HEAT MAP FUNCTION FOR CORRELATIONS

def heat_map_corrmatr(dataset,a):
    A = np.corrcoef(dataset[:,2:].T)
    n_col = A.shape[1]

    features = np.arange(0,dataset.shape[1])
    # Set the size of the fihure
    fig, ax = plt.subplots(figsize=(max(0.5*n_col,15) , max(0.5*n_col,15)))
    im = ax.imshow(A, cmap = 'Blues')
    # Assign the dimensions and the ticks lables of the x and y axes
    ax.set_xticks(np.arange(len(features[2:])))
    ax.set_yticks(np.arange(len(features[2:])))

    ax.set_xticklabels(features[2:], fontsize=12)
    ax.set_yticklabels(features[2:], fontsize=12)

   
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

    # Fill in each cell of the correlation matrix withnthe respective value
    for i in range(n_col):
        for j in range(n_col):
            text = ax.text(j, i, round(A[i, j], 2),ha="center", va="center", color="w", fontsize=12)
    if a==2:
       ax.set_title("Heatmap for the correlations of the features, Jet {name}".format(name = [a,a+1] ) , fontsize=20)
    else:
       ax.set_title("Heatmap for the correlations of the features, Jet {name}".format(name = a ) , fontsize=20) 
        
    fig.tight_layout()
    plt.colorbar(im)
    plt.show()

#%% Function to visualize and compare empirical distribution according to the label


def Plot_empirical_distributions(y, tx, train):
    
    features = np.arange(0, train.shape[1]-2)
    
    for i in range(len(features)):
        
       # Select the indexes corresponding to the lable -1 and the ones corresponding to the lable 1
        index_P = [y==1][0]
        index_N = [y==0][0] # we use 0 as we substitute the -1 with 0 when we train
           
        plt.hist(tx[index_P,i] ,100, histtype ='step',color='b',label='y == 1',density=True)      
        plt.hist(tx[index_N,i] ,100, histtype ='step',color='r',label='y == -1',density=True)  
        plt.legend(loc = "upper right")
        plt.title("{name}, Feature: {id}/{tot}".format(name=features[i],id=i,tot=len(features)-1), fontsize=12)
        plt.show()    
   
    
#%% Fuction for feature inspection: Plots each of the feature and the corresponding summary statistics

def Plot_Features_and_stats(dataset,a):
    features = np.arange(0,dataset.shape[1]-2)
# Create a plot to visualize the magnitude of each observation of a feature and spot possible outliers
    for i in range(len(features)):
        xlab = np.arange(len(dataset))
        fig = plt.figure()
        plt.plot(xlab,dataset[:,i],'g')
        plt.title(features[i], fontsize=12)
        plt.xlabel('rows',fontsize=12)
        
        # Calcuòate summary statistics
        mean = np.round_(np.mean(dataset[:,i]),decimals = 1)
        median =  np.round_(np.median(dataset[:,i]),decimals = 1)
        std =  np.round_(np.std(dataset[:,i]),decimals = 1)
        skew =  np.round_(3*(mean - median)/std,decimals = 1)
        
        # Print ordered summary statistics
        if a==2:
            print('Summary statistics of the feature {featurename}, Jet {jname}'.format(featurename = features[i],jname = [a,a+1]))
        else:
            print('Summary statistics of the feature {featurename}, Jet {jname}'.format(featurename = features[i],jname = a))
        
        print('_________________________________________________________________')    
        d = {mean: [median, std, skew]}
        print ("{:<8} {:<15} {:<20} {:<10}".format('Mean','Median','St. Deviation','Skewness'))
        for k, v in d.items():
            l, p, c = v
            print ("{:<8} {:<15} {:<20} {:<10}".format(k, l, p, c))
        print('#################################################################')    
        plt.show()
