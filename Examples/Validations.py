import numpy as np
import matplotlib.pyplot as plt
from Tools.Utils import Utils
from Tools.AdditionalProcessing import AdditionalProcessing
from Tools.Crossvalidation import regular_CV, leave_maps_out_CV, accuracy_by_maps, leave_one_sample_out_CV
from sklearn.svm import SVC
from skelarn.model_selection import StratifiedShuffleSplit


uti = Utils()
addprep = AdditionalProcessing()

##===================================================Start Running===========================================================##
##------------------------------------------------input data dictionary------------------------------------------------------##

data_dict_path = r'E:\Xie group\DataAnalysis\datadict\U18\functionalization\data10_18_21_FuncCovidVariants_Batch3_subed.npy'
data_matrix, label, group, sample_name, raman_shift, map_index = uti.input(data_dict_path)

##---------------------------------------------------preprocessing-----------------------------------------------------------##

X = uti.smoothing(data_matrix)

##=================================================Cross Validations=========================================================##
## Run only one cross validation per time depending on your study
##-------------------------------------------------define classifiers--------------------------------------------------------##

splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
clf_svc = SVC(C=5.0, scale='gamma') # you can tune 'C' to optimize your performance

##----------------------------------------------regular cross validation-----------------------------------------------------##

X = data_matrix
y = group
regular_CV(X, y, splitter, output='accuracy')

##------------------------------------------------LMsO cross validation------------------------------------------------------##
## Split the total maps into K folds for cross validation

# X = data_matrix
# y = group
# leave_maps_out_CV(X, y, map_index, k=5) # use appropriate K according to your dataset


##-------------------------------------------------LOSO cross validation-----------------------------------------------------##
## Split the total maps into K folds for cross validation

# X = data_matrix
# y = group
# leave_one_sample_out_CV(X, y, label, map_index, 0.5)
