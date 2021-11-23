import numpy as np
import matplotlib.pyplot as plt
from Tools.Utils import Utils
from Tools.AdditionalProcessing import AdditionalProcessing
from Tools.CrossValidation import regular_CV, leave_maps_out_CV, accuracy_by_maps, leave_one_sample_out_CV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit


uti = Utils()
addprep = AdditionalProcessing()

##===================================================Start Running===========================================================##
##------------------------------------------------input data dictionary------------------------------------------------------##

data_dict_path = r'/Users/tieyili/Desktop/data11_10_21_SpikedSalivaBlindTest_2ndRoundTest_Sub.npy'
data_matrix, label, group, sample_name, raman_shift, map_index = addprep.input_data(data_dict_path)

##---------------------------------------------------preprocessing-----------------------------------------------------------##

data_matrix = uti.smoothing(data_matrix)

##=================================================Cross Validations=========================================================##
## Run only one cross validation per time depending on your study
##-------------------------------------------------define classifiers--------------------------------------------------------##

clf_svc = SVC(C=5.0, gamma='scale') # you can tune 'C' to optimize your performance

#-----------------------------------------------regular cross validation-----------------------------------------------------##

splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
X = data_matrix
y = label
regular_CV(clf_svc, X, y, splitter, output='accuracy')

##------------------------------------------------LMsO cross validation------------------------------------------------------##
## Split the total maps into K folds for cross validation

# X = data_matrix
# y = group
# leave_maps_out_CV(clf_svc, X, y, map_index, k=5) # use appropriate K according to your dataset


##-------------------------------------------------LOSO cross validation-----------------------------------------------------##
## Only use leave-one-sample-out cross validation for patient/healthy and each group contains more than one samples

# X = data_matrix
# y = group
# pred, true, score = leave_one_sample_out_CV(clf_svc, X, y, label, map_index, 0.5)
# addprep.manual_roc(true, score)

##-------------------------------------------------LPSsO cross validation-----------------------------------------------------##
## Only use leave-pair-of-samples-out cross validation for patient/healthy and each group contains more than one samples

# X = data_matrix
# y = group
# pred, true, score = leave_pair_of_samples_out_CV(clf_svc, X, y, label, group_dict, map_index, 0.5)
# addprep.manual_roc(true, score)
