import numpy as np
import matplotlib.pyplot as plt
from Tools.Utils import Utils
from Tools.AdditionalProcessing import AdditionalProcessing
from sklearn.svm import SVC


uti = Utils()
addprep = AdditionalProcessing()

##==========================================================Start Running====================================================================##
##--------------------------------------------------input training data dictionary-----------------------------------------------------------##

training_data_dict_path = r'E:\Xie group\DataAnalysis\datadict\U18\functionalization\data10_18_21_FuncCovidVariants_Batch3_subed.npy'
X_training, label_training, group_training, sample_name_training, raman_shift, map_index_training = addprep.input_data(training_data_dict_path)

##----------------------------------------------------input test data dictionary-------------------------------------------------------------##

test_data_dict_path = r'E:\Xie group\DataAnalysis\datadict\U18\functionalization\data10_18_21_FuncCovidVariants_Batch3_subed.npy'
X_test, label_test, _, sample_name_test, _, map_index_test = addprep.input_data(test_data_dict_path)

##----------------------------------------------------------preprocessing--------------------------------------------------------------------##

X_training = uti.smoothing(X_training)
X_test = uti.smoothing(X_test)

##--------------------------------------------------------define classifier------------------------------------------------------------------##

clf_svc = SVC(C=5.0, gamma='scale') # use the model as well as parameters giving the best cross validation performance

##-------------------------------------------------------generate predictions----------------------------------------------------------------##

clf_svc.fit(X_training, group_training)
predictions = clf_svc.predict(X_test)
# addprep.test_summary_spec_base(label_test, predictions) # use this line for one spectrum per vesicle/particle
# addprep.test_summary_map_base(label_test, map_index_test, predictions) # use this line for more than one spectra per vesicle/particle
