import numpy as np

############################## PRE - PROCESSING ###############################

def split_jet(dataset, col24):
    """ Split the dataset according the categorical values of col24 'PRI_jet_num'
        Args:
            dataset: shape=(N, 32) (N number of events)
            col24: shape=(N, ) (24th column of dataset, 22nd feature)
        Return: 
            mylist: list[list_a, list_b, list_c]
                    every internal list refers to a categorical value and contains:
                        - the sub-dataset without all the zero-variance features
                        - the indexes of the events in the sub-dataset
                        - the indexes of the columns in the sub-dataset """
                        
    col_indexes = np.arange(dataset[0,:].size)
    
    row_indexes = np.arange(dataset[:,0].size)
    ind0 = row_indexes[col24==0]
    ind1 = row_indexes[col24==1]
    ind23 = row_indexes[(col24==2) | (col24==3)]
    
    jet_0 = dataset[ind0,:]
    jet_1 = dataset[ind1,:]
    jet_23 = dataset[ind23,:]
    
    mylist = []
    mydata = [jet_0, jet_1, jet_23]
    myind = [ind0, ind1, ind23]
    
    for i in range(len(mydata)):
        col_index_to_keep = col_indexes[np.std(mydata[i],axis=0) != 0]
        mydata[i] = mydata[i][:,col_index_to_keep]
        internal_list = [mydata[i], myind[i], col_index_to_keep]
        mylist.append(internal_list)
        
    return mylist