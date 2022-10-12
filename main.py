import sys
sys.path.append("../../ML_project1")

import helpers
from helpers import *

#%% Importing the dataset

train = load_dataset()

print(train.shape)
print(train[0:3,:])

#%%
print(count_999_for_feature(train))
