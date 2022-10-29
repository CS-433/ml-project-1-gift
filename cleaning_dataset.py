import numpy as np

############################# CLEANING DATA ###################################

def ind_pos_neg(dataset):
    """ Returns two boolean arrays whose True values indicate whether the
        corresponding feature has all non-negative values and all non-positive
        values, respectively
        Args:
            dataset: shape=(N, D+2) (N number of events, D number of features)
        Returns:
            ind_pos: shape=(D+2, )
            ind_neg: shape=(D+2, ) """
            
    n_cols = dataset[0,:].size
    ind_col = np.arange(n_cols)
    ret_pos = np.array([np.count_nonzero(dataset[:,j] < 0) for j in ind_col])
    ind_pos = (ret_pos == 0)
    ret_neg = np.array([np.count_nonzero(dataset[:,j] > 0) for j in ind_col])
    ind_neg = (ret_neg == 0)
    
    return ind_pos, ind_neg

def count_nan(dataset):
    """ Count the -999 values for each feature in the dataset
        Args:
            dataset: shape=(N, D+2) (N number of events, D number of features)
        Returns:
            ret: shape=(D, ) """
    
    n_cols = dataset[0,:].size
    ind_col = np.arange(n_cols)
    ret = np.array([np.count_nonzero(dataset[:,j]==-999) for j in ind_col])
    
    return ret

def count_nan_for_feature(col_vector):
    """ Count the -999 values for a given feature
        Args:
            dataset: shape=(N, D+2) (N number of events, D number of features)
        Returns:
            ret: scalar """
            
    return np.count_nonzero(col_vector==-999)
    
def remove_feature_with_50pec(dataset):
    """ Remove all the features in the dataset having more than the 50% 
        of -999 values 
        Args:
            dataset: shape=(N, D+2) (N number of events, D number of features)
        Returns:
            ret: shape=(N, D+2-d) (d number of removed features)"""
            
    n = dataset[:, 0].size
    cnt = count_nan_for_feature(dataset)
    feat_to_keep = (cnt < 0.5*n)
    ret = dataset[:, feat_to_keep].copy()
    
    return ret

def remove_nan_rows(dataset):
    """ Remove all the events in the dataset with at least one -999 value 
        among the features
        Args:
            dataset: shape=(N, D+2) (N number of events, D number of features)
        Returns:
            ret: shape=(N-n, D+2) (n number of removed events)
            selec_rows: shape=(N-n, )
        """
        
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
        feature in greater than 2 standard deviations) 
        Args:
            dataset: shape=(N, D+2) (N number of events, D number of features)
        Returns:
            ret: shape=(N-n, D+2) (n number of removed events)
            ind_to_keep: shape=(N-n, )"""
            
    ret = dataset.copy()
    
    mean = np.mean(ret[:,2:], axis=0)
    std = np.std(ret[:,2:], axis=0)
    
    ind_to_keep = []
    
    for i in np.arange(len(dataset[:,0])):
        if(np.count_nonzero(dataset[i, 2:] - mean > 3*std) == 0):
            ind_to_keep.append(i)
    ind_to_keep = np.array(ind_to_keep)
    ind_to_keep = ind_to_keep.astype(int)
    
    return ret[ind_to_keep], ind_to_keep

def count_outliers(dataset):
    """ Count the amount of outliers for each feature
        ('outlier': value whose distance from the mean of the corresponding 
        feature in greater than 3 standard deviations) 
        Args:
            dataset: shape=(N, D+2) (N number of events, D number of features)
        Returns:
            ret: shape=(D, ) """
            
    ret = dataset.copy()
    
    mean = np.mean(ret[:,2:], axis=0)
    std = np.std(ret[:,2:], axis=0)
    
    cnt_vec = np.array([np.count_nonzero(ret[:,i]-mean[i-2] > 3*std[i-2]) for i in np.arange(2,31)] )
    
    return cnt_vec

def fix_outliers(dataset):
    """ Cap the outliers, converting evry outlier with the corresponding critical
        value of mean+2*std or mean-2*std
        ('outlier': value whose distance from the mean of the corresponding 
        feature in greater than 3 standard deviations)
        Args:
            dataset: shape=(N, D+2) (N number of events, D number of features)
        Returns:
            ret: shape=(N, D+2)
        """
        
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
    """ Return the indexes of the events whose -999 values are greater than tol
        Args:
            dataset: shape=(N, D+2) (N number of events, D number of features)
            tol: scalar(int)
        Returns:
            ind_nan_r: shape=(N, ) """
            
    check = []
    for i in range(len(dataset[:,0])):
        check.append(np.count_nonzero(dataset[i,:]==-999))

    check  = np.array(check)   
    ind_nan_r =  check > tol
    
    return ind_nan_r

def balance(dataset):
    """ Balance the dataset obtaining half of y=0 and the other half y=1
        The excess events are removed according to the amount of -999 values:
        the higher the -999 values for a given event, the earlier it is removed 
        Args:
            dataset: shape=(N, D+2) (N number of events, D number of features)
        Returns:
            ret: shape=(2*n, D+2) (n number of the less frequent prediction) """
            
    num_ones = np.sum(dataset[:,1]==1)
    num_minus_ones = np.sum(dataset[:,1]== 0)
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

