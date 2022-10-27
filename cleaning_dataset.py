import numpy as np

def count_nan(dataset):
    """ Count the -999 values for each feature in the dataset """
    n_cols = dataset[0,:].size
    ind_col = np.arange(n_cols)
    ret = np.array([np.count_nonzero(dataset[:,j]==-999) for j in ind_col])
    
    return ret

def count_nan_for_feature(col_vector):
    """ Count the -999 values for a given feature """
    return np.count_nonzero(col_vector==-999)
    
def remove_feature_with_50pec(dataset):
    """ Remove all the features in the dataset having more than the 50% 
        of -999 values """
    n = dataset[:, 0].size
    cnt = count_nan_for_feature(dataset)
    feat_to_keep = (cnt < 0.5*n)
    ret = dataset[:, feat_to_keep].copy()
    
    return ret

def remove_deriv(dataset):
    """ Remove the 'DER' features in the dataset """
    feat_to_keep = np.array([0, 1, 15, 16, 17, 18, 19, 20, 21, 22,
                             23, 24, 25, 26, 27, 28, 29, 30])
    ret = dataset[:, feat_to_keep].copy()
    
    return ret

def remove_nan_rows(dataset):
    """ Remove all the events in the dataset with at least one -999 value 
        among the features """
    check = []
    selec_rows = []

    for i in range(len(dataset[:,0])):
        check = np.count_nonzero(dataset[i,:]==-999)
        if(check==0):
            selec_rows.append(i)
        
    ret = dataset[selec_rows].copy()
    selec_rows = np.array(selec_rows)
        
    return ret, selec_rows

def remove_outlier_rows(dataset):
    """ Remove all the events in the dataset with at least one outlier
        among the features
        ('outlier': value whose distance from the mean of the corresponding 
        feature in greater than 3 standard deviations) """
    ret = dataset.copy()
    
    mean = np.mean(ret[:,2:], axis=0)
    std = np.std(ret[:,2:], axis=0)
    
    ind_to_keep = []
    
    for i in np.arange(len(dataset[:,0])):
        if(np.count_nonzero(dataset[i, 2:] - mean > 3*std) == 0):
            ind_to_keep.append(i)
    ind_to_keep = np.array(ind_to_keep)
    ind_to_keep = ind_to_keep.astype(int)
    
    return ret[ind_to_keep]

def count_outliers(dataset):
    """ Count the amount of outliers for each feature
        ('outlier': value whose distance from the mean of the corresponding 
        feature in greater than 3 standard deviations) """
    ret = dataset.copy()
    
    mean = np.mean(ret[:,2:], axis=0)
    std = np.std(ret[:,2:], axis=0)
    
    cnt_vec = np.array([np.count_nonzero(ret[:,i]-mean[i-2] > 3*std[i-2]) for i in np.arange(2,31)] )
    
    return cnt_vec

def fix_outliers(dataset):
    ret = dataset[:,2:].copy()
    first_col = dataset[:,:2].copy()
    row = ret.shape[0]
    cols = ret.shape[1]
    
    mean = np.mean(ret,axis=0)
    std = np.std(ret,axis=0)
    
    for i in range(row):
        for j in range(cols):
            if ret[i,j] > (mean[j] + 2*std[j]):
                ret[i,j] = (mean[j] + 2*std[j])
            if ret[i,j] < (mean[j] - 2*std[j]):
                ret[i,j] = (mean[j] - 2*std[j])
                
    final = np.c_[first_col, ret]
    
    return final

def nan_row_index(dataset, tol):
    """ Return the indexes of the events whose -999 values are greater than tol """
    check = []
    for i in range(len(dataset[:,0])):
        check.append( np.count_nonzero(dataset[i,:]==-999))

    check  = np.array(check)   
    ind_nan_r =  check > tol
    
    return ind_nan_r

def balance(dataset):
    """ Balance the dataset obtaining half of y=0 and the other half y=1
        The excess events are removed according to the amount of -999 values:
        the higher the -999 values for a given event, the earlier it is removed """
    num_ones = np.sum(dataset[:,1]==1)
    num_minus_ones = np.sum(dataset[:,1]== 0)
    Dim = num_ones + num_minus_ones
    ret = dataset.copy()
    
    index_minus_ones = dataset[:,1] == 0 
    index_ones = dataset[:,1] == 1
    tol = 0
    found = False
    
    if (num_minus_ones > num_ones) :
        diff = num_minus_ones - num_ones
        index = index_minus_ones
    else:
        diff = num_ones - num_minus_ones
        index = index_ones
        
    while not found:
        ind_nan_rows = nan_row_index(ret, tol)
        to_delete = ind_nan_rows * index
        n = np.count_nonzero(to_delete==True)
        
        if n < diff:
            found = True
        else:
            tol = tol + 1
         
    to_keep = np.array([not to_delete[i] for i in range(len(to_delete))])
    ret = ret[to_keep,:]
        
    index = index[to_keep]
    tol = tol  - 1
    ind_nan = nan_row_index(ret, tol)
        
    to_del = ind_nan * index
        
    cumsum = np.cumsum(to_del)
    ind = np.min(np.where(cumsum==diff-n))
        
    to_del[ind:] = False
    to_k = np.array([not to_del[i] for i in range(len(to_del))])
    ret = ret[to_k,:]

    return ret




def count_neg(dataset):
    """ Count the negative values for each feature in the dataset """
    n_cols = dataset[0,:].size
    ind_col = np.arange(n_cols)
    ret = np.array([np.count_nonzero(dataset[:,j] < 0) for j in ind_col])
    
    return ret
