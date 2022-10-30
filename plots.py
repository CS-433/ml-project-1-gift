import numpy as np
import matplotlib.pyplot as plt
from utilities_linear_regression import *
from utilities_logistic_regression import *
from feature_engineering import *

#%%################## FEATURE INSPECTION PLOTS FUNCTIONS ######################
def get_feature_names():
    
    names = np.array(['Id','Prediction','DER_mass_MMC',
                          'DER_mass_transverse_met_lep','DER_mass_vis',
                          'DER_pt_h','DER_deltaeta_jet_jet','DER_mass_jet_jet',
                          'DER_prodeta_jet_jet','DER_deltar_tau_lep',
                          'DER_pt_tot','DER_sum_pt','DER_pt_ratio_lep_tau',
                          'DER_met_phi_centrality','DER_lep_eta_centrality',
                          'PRI_tau_pt','PRI_tau_eta','PRI_tau_phi',
                          'PRI_lep_pt','PRI_lep_eta','PRI_lep_phi',
                          'PRI_met','PRI_met_phi','PRI_met_sumet',
                          'PRI_jet_leading_pt','PRI_jet_leading_eta',
                          'PRI_jet_leading_phi','PRI_jet_subleading_pt',
                          'PRI_jet_subleading_eta','PRI_jet_subleading_phi',
                          'PRI_jet_all_pt'])
    return names

# Fuction for feature inspection: Plots each of the feature and the corresponding summary statistics
def plot_features_and_stats(dataset, a, labels):
    features = np.arange(0,dataset.shape[1]-2)
    # Create a plot to visualize the magnitude of each observation of a feature and spot possible outliers
    for i in range(len(features)):
        name = labels[i+2]
        xlab = np.arange(len(dataset))
        fig = plt.figure()
        plt.plot(xlab, dataset[:,i],'g')
        plt.title(name, fontsize=12)
        plt.xlabel('rows',fontsize=12)
        
        # Calcuòate summary statistics
        mean = np.round_(np.mean(dataset[:,i]),decimals = 1)
        median =  np.round_(np.median(dataset[:,i]),decimals = 1)
        std =  np.round_(np.std(dataset[:,i]),decimals = 1)
        skew =  np.round_(3*(mean - median)/std,decimals = 1)
        
        # Print ordered summary statistics
        if a==2:
            print('Summary statistics of the feature {featurename}, Jet {jname}'.format(featurename = name, jname = [a,a+1]))
        else:
            print('Summary statistics of the feature {featurename}, Jet {jname}'.format(featurename = name, jname = a))
        
        print('_________________________________________________________________')    
        d = {mean: [median, std, skew]}
        print ("{:<8} {:<15} {:<20} {:<10}".format('Mean','Median','St. Deviation','Skewness'))
        for k, v in d.items():
            l, p, c = v
            print ("{:<8} {:<15} {:<20} {:<10}".format(k, l, p, c))
        print('#################################################################')    
        plt.show()

# Heatmap correlation matrix for the features
def heat_map_correlation_matrix(dataset, a, labels):
    A = np.corrcoef(dataset[:,2:].T)
    n_col = A.shape[1]

    features = np.arange(0, dataset.shape[1])
    
    # Set the size of the figure
    fig, ax = plt.subplots(figsize=(max(0.5*n_col,15) , max(0.5*n_col,15)))
    im = ax.imshow(A, cmap = 'Blues')
    
    # Assign the dimensions and the ticks lables of the x and y axes
    ax.set_xticks(np.arange(len(features[2:])))
    ax.set_yticks(np.arange(len(features[2:])))

    ax.set_xticklabels(labels[2:], fontsize=12)
    ax.set_yticklabels(labels[2:], fontsize=12)

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

# Function to visualize and compare empirical distribution according to the label
def plot_empirical_distributions(y, tx, labels, jet):
    
    features = np.arange(tx.shape[1])
    
    for i in range(len(features)):
        
       # Select the indexes corresponding to the lable -1 and the ones corresponding to the lable 1
        index_P = [y == 1][0]
        index_N = [y == 0][0]
           
        plt.hist(tx[index_P, i], 100, histtype='step', color='g', label='y == 1', density=True)      
        plt.hist(tx[index_N, i], 100, histtype='step', color='m', label='y == -1', density=True)  
        plt.legend(loc = "upper right")
        plt.title("jet={jet}, {name}, Feature: {id}/{tot}".format(jet=jet, name=labels[i+2], id=i+1, tot=len(features)), fontsize=12)
        plt.show()
        
#%%############################ PLOT OF LAMBDAS ###############################
def plot_lambdas(train_errors, test_errors, lambdas, degree):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set
    
    degree is just used for the title of the plot.
    """
    
    for i in range(len(train_errors)):
        if i==0:
            plt.semilogx(lambdas, train_errors[i], color='b', marker='*', label="Train error")
            plt.semilogx(lambdas, test_errors[i], color='r', marker='*', label="Test error")
        else:
            plt.semilogx(lambdas, train_errors[i], color='b', marker='*')
            plt.semilogx(lambdas, test_errors[i], color='r', marker='*')
    
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    plt.title("Polynomial expansion for degree " + str(degree))
    leg = plt.legend(loc="upper left", shadow=True)
    leg.draw_frame(False)
    plt.savefig("lambdas_for_RR_{deg}".format(deg=degree))
    
#%%########################## ACCURACY STEP BY STEP ###########################
def accuracy_fun(true,pred):
    #Given 2 vectors, returns the accuracy. To be used for OLS/ridge
    index_neg = (pred< 0)
    index_pos = (pred>= 0)
    
    pred[index_neg] = -1
    pred[index_pos] = 1
    
    correct = 0
    
    for k in range(len(true)):
        if true[k] == pred[k]:
            correct = correct+1
            
    accu = correct/len(true)
    return accu

def accuracy_fun_log(true,pred):
    #Given 2 vectors, returns the accuracy. To be used for logistic
    index_neg = (pred < 0.5)
    index_pos = (pred >= 0.5)
    
    pred[index_neg] = 0
    pred[index_pos] = 1
    
    correct = 0
    
    for k in range(len(true)):
        if true[k] == pred[k]:
            correct = correct+1
            
    accu = correct/len(true)
    return accu

def cross_valid_train_full(dataset):
    #train accuracy of OLS on the entire original dataset
    accuracy = np.zeros(100)
    
    true = dataset[:,1]
    l = len(dataset)
    train = dataset[:,2:]
    
    for i in range(100):
        perm = np.random.permutation(l)
        train_index = perm[:int(np.floor(l*0.8))]
        test_index = perm[int(np.floor(l*0.8)):]
        train_i = train[train_index,:]
        test_i = train[test_index,:]
        true_train = true[train_index]
        true_test = true[test_index]
        ws = least_squares(true_train,train_i)[0]
        pred = test_i.dot(ws)
        accu = accuracy_fun(true_test,pred)
        accuracy[i] = accu
        
    return accuracy, np.mean(accuracy)

def cross_valid_train_ols_splitted(datasets):
    #train accuracy of OLS 
    accuracy = np.zeros(100)
    for i in range(100):
        true = []
        pred = []
        
        for j in range(len(datasets)):
            l = len(datasets[j])
            perm = np.random.permutation(l)
            train_index = perm[:int(np.floor(l*0.8))]
            test_index = perm[int(np.floor(l*0.8)):]
            train_i = datasets[j][train_index,2:]
            true_train = datasets[j][train_index,1]
            test_i = datasets[j][test_index,:]
            true.append(test_i[:,1])
            ws = least_squares(true_train,train_i)[0]
            pred_i = test_i[:,2:].dot(ws)
            pred.append(pred_i)
            
        true_vec = np.hstack((true[0],true[1],true[2]))
        pred_vec = np.hstack((pred[0],pred[1],pred[2]))
        accu = accuracy_fun(true_vec,pred_vec)
        accuracy[i] = accu
        
    return accuracy, np.mean(accuracy)

def cross_valid_train_ridge_splitted(datasets):
    #train accuracy of ridge
    accuracy = np.zeros(100)
    lambdas = np.logspace(-10, 0, 30)
    k_fold = 10
    lambda_vec = []
    for i in range(len(datasets)):
        ids, y, tx = split_into_ids_y_tx(datasets[i])
        l = cross_validation_demo_tx_lin(y, tx, k_fold, lambdas)[0]
        lambda_vec.append(l)
        
    for i in range(100):
        true = []
        pred = []
        
        for j in range(len(datasets)):
            l = len(datasets[j])
            perm = np.random.permutation(l)
            train_index = perm[:int(np.floor(l*0.8))]
            test_index = perm[int(np.floor(l*0.8)):]
            train_i = datasets[j][train_index,2:]
            true_train = datasets[j][train_index,1]
            test_i = datasets[j][test_index,:]
            true.append(test_i[:,1])
            ws = ridge_regression(true_train, train_i, lambda_vec[j])[0]
            pred_i = test_i[:,2:].dot(ws)
            pred.append(pred_i)
            
        true_vec = np.hstack((true[0],true[1],true[2]))
        pred_vec = np.hstack((pred[0],pred[1],pred[2]))
        accu = accuracy_fun(true_vec,pred_vec)
        accuracy[i] = accu
        
    return accuracy, np.mean(accuracy)

def mega_cross_path_train(original,test_original,col24_train,col24_test):
    
    accuracies = []
    means = []
    
    accu_ols_full, mean_ols_full = cross_valid_train_full(original)
    
    accuracies.append(accu_ols_full)
    means.append(mean_ols_full)
    
    #STEP2: splitted and filled column 3
    train_datasets = split_jet(original, col24_train)
    test_datasets = split_jet(test_original, col24_test)
    for i in range(len(test_datasets)):
        n_rows = test_datasets[i][0][:,0].size
        col = np.zeros(n_rows)
        test_datasets[i][0] = np.insert(test_datasets[i][0], 1, col, axis=1)
    
    k_fold = 5
    lambdas = np.logspace(-10, 0, 30)

    train_filled = []
    test_filled = []

    for i in range(len(train_datasets)):
        train_jet_filled, test_jet_filled = feature_regression_col3(train_datasets[i][0], test_datasets[i][0], k_fold, lambdas, seed = 1)
        train_filled.append(train_jet_filled)
        
    accu_ols_split,mean_ols_split = cross_valid_train_ols_splitted(train_filled)
    accuracies.append(accu_ols_split)
    means.append(mean_ols_split)
    
    #STEP3: correlated columns and outliers
    train_filled[0] = np.delete(train_filled[0], [4,5,8], 1)
    train_filled[1] = np.delete(train_filled[1], [4,5,8,20], 1)
    train_filled[2] = np.delete(train_filled[2], [4,8,11,30], 1)
    
    train_out = []
    for i in range(len(train_filled)):
        tr = fix_outliers(train_filled[i])
        train_out.append(tr)
        
    accu_ols_out, mean_ols_out = cross_valid_train_ols_splitted(train_out)
    accuracies.append(accu_ols_out)
    means.append(mean_ols_out)
        
    #STEP4: transformations and expansion to 7
    indexes_skew = []
    train_skew = []
    
    for i in range(len(train_out)):
        ind_train = count_neg(train_out[i]) == 0
        ind_train[:2] = False
        tra = correct_skewness(train_out[i], ind_train)
        train_skew.append(tra)
    
    angle_cols0 = np.array([7, 10, 13, 15])
    angle_cols1 = np.array([7, 10, 13, 15, 18])
    angle_cols23 = np.array([10, 14, 17, 19, 23, 26])
    to_correct = [angle_cols0, angle_cols1, angle_cols23]
    train_angle = train_skew
    
    for i in range(len(train_skew)):
        train_angle[i][:, to_correct[i]] = np.cos(train_skew[i][:, to_correct[i]])
    
    train_exp = []
    deg = 7

    for i in range(len(train_angle)):
        curr_exp_train = poly_expansion_blind(train_angle[i], deg)
        train_exp.append(curr_exp_train)
        
    accu_ols_exp_7,mean_ols_exp_7=cross_valid_train_ols_splitted(train_exp)
    accuracies.append(accu_ols_exp_7)
    means.append(mean_ols_exp_7)
    
    #STEP5: add prods
    train_prod = []
    for i in range(len(train_angle)):
        tr = feature_cross_products(train_angle[i])
        train_prod.append(np.c_[train_exp[i], tr])
        
    accu_ols_prod, mean_ols_prod=cross_valid_train_ols_splitted(train_prod)
    accuracies.append(accu_ols_prod)
    means.append(mean_ols_prod)
      
    # STEP6: add sqrt
    train_sqrt = []
    for i in range(len(train_angle)):
        tr = squareroot(train_angle[i])
        train_sqrt.append(np.c_[train_prod[i], tr])
        
    accu_ols_sqrt,mean_ols_sqrt=cross_valid_train_ols_splitted(train_sqrt)
    accuracies.append(accu_ols_sqrt)
    means.append(mean_ols_sqrt)
     
    #STEP7: ridge
    accu_ridge, mean_ridge = cross_valid_train_ridge_splitted(train_sqrt)
    accuracies.append(accu_ridge)
    means.append(mean_ridge)
    
    return accuracies,means

def cross_valid_train_logistic_splitted(datasets):
    #train accuracy of logistic
    for i in range(3):
        index_train = (datasets[i][:,1] == -1)
        datasets[i][index_train, 1] = 0
    
    accuracy = np.zeros(5)
    
    for i in range(5):
        true = []
        pred = []
        
        for j in range(len(datasets)):
            l = len(datasets[j])
            perm = np.random.permutation(l)
            train_index = perm[:int(np.floor(l*0.8))]
            test_index = perm[int(np.floor(l*0.8)):]
            train_i = datasets[j][train_index,2:]
            true_train = datasets[j][train_index,1]
            test_i = datasets[j][test_index,:]
            true.append(test_i[:,1])
            ws = logistic_regression_gradient_descent_demo(true_train, train_i)[1]
            pred_i = test_i[:,2:].dot(ws)
            pred.append(pred_i)
            
        true_vec = np.hstack((true[0],true[1],true[2]))
        pred_vec = np.hstack((pred[0],pred[1],pred[2]))
        accu = accuracy_fun_log(true_vec,pred_vec)
        accuracy[i] = accu
        
    return accuracy, np.mean(accuracy)

def cross_valid_train_GD_splitted(datasets):
    #train accuracy of GD
    accuracy = np.zeros(100)
    max_iters = 1000
    gamma = 0.05
    w_0 = []
    w_0.append(np.ones(datasets[0][:,2:].shape[1]))
    w_0.append(np.ones(datasets[1][:,2:].shape[1]))
    w_0.append(np.ones(datasets[2][:,2:].shape[1]))
    for i in range(5):
        true = []
        pred = []
        
        for j in range(len(datasets)):
            l = len(datasets[j])
            perm = np.random.permutation(l)
            train_index = perm[:int(np.floor(l*0.8))]
            test_index = perm[int(np.floor(l*0.8)):]
            train_i = datasets[j][train_index,2:]
            true_train = datasets[j][train_index,1]
            test_i = datasets[j][test_index,:]
            true.append(test_i[:,1])
            ws = gradient_descent(true_train, train_i, w_0[j], max_iters, gamma)[1][1000]
        
            print(ws)
            pred_i = test_i[:,2:].dot(ws)
            pred.append(pred_i)
            
        true_vec = np.hstack((true[0],true[1],true[2]))
        pred_vec = np.hstack((pred[0],pred[1],pred[2]))
        accu = accuracy_fun(true_vec,pred_vec)
        accuracy[i] = accu
        
    return accuracy, np.mean(accuracy)

def cross_valid_train_SGD_splitted(datasets):
    #train accuracy of SGD 
    accuracy = np.zeros(100)
    max_iters = 100
    gamma = 0.05
    batch_size = 10
    w_0 = []
    w_0.append(np.ones(datasets[0][:,2:].shape[1]))
    w_0.append(np.ones(datasets[1][:,2:].shape[1]))
    w_0.append(np.ones(datasets[2][:,2:].shape[1]))
    for i in range(5):
        true = []
        pred = []
        
        for j in range(len(datasets)):
            l = len(datasets[j])
            perm = np.random.permutation(l)
            train_index = perm[:int(np.floor(l*0.8))]
            test_index = perm[int(np.floor(l*0.8)):]
            train_i = datasets[j][train_index,2:]
            true_train = datasets[j][train_index,1]
            test_i = datasets[j][test_index,:]
            true.append(test_i[:,1])
            ws = stochastic_gradient_descent(true_train, train_i, w_0[j],batch_size, max_iters, gamma)[1][100]
        
            print(ws)
            pred_i=test_i[:,2:].dot(ws)
            pred.append(pred_i)
            
        true_vec = np.hstack((true[0],true[1],true[2]))
        pred_vec = np.hstack((pred[0],pred[1],pred[2]))
        accu = accuracy_fun(true_vec,pred_vec)
        accuracy[i] = accu
        
    return accuracy, np.mean(accuracy)