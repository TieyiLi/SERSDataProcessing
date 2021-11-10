import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from AffinityMatrix import distance_matrix
from Utils import Utils
from AdditionalProcessing import AdditionalProcessing


uti = Utils()
addprep = AdditionalProcessing()

##===================================================Start Running===========================================================##
##------------------------------------------------input data dictionary------------------------------------------------------##

data_dict_path = r'E:\Xie group\DataAnalysis\datadict\U18\functionalization\data10_18_21_FuncCovidVariants_Batch3_subed.npy'
data_matrix, label, group, sample_name, raman_shift, map_index = uti.input(data_dict_path)

##----------------------------------------------------proprecessing----------------------------------------------------------##
## noise reduction or smoothing
X = uti.smoothing(data_matrix)

##------------------------------------------------compute distance matrix----------------------------------------------------##
## this block always requires a few minutes, therefore you can save the dist_mtx in a (.npy) file then load it by uncommenting
## the next lines after 'dist_mtx = distance_matrix(X)' to save running time

dist_mtx = distance_matrix(X) 
# np.save(r'E:\Xie group\DataAnalysis\DistanceMatrix\data9_22_21_funked_covid_variants_subed.npy', dist_mtx)

##---------------------------------------------------clustering by HCA-------------------------------------------------------##
## once you have the predictions by clustering, you can proceed to the following analysis that you want, for example searching
## for mixed clusters, relabeling data and so on.

# dist_mtx = np.load(r'E:\Xie group\DataAnalysis\clustering labels\data9_22_21_funked_covid_variants_subed.npy')
HCA = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='complete', distance_threshold=0.125)
clustering_predictions = HCA.predict(dist_mtx)

##---------------------------------------------------followed by analysis----------------------------------------------------##
## the following is an example of showing the spectra belonging to every cluster in order to check clustering result. Watch your
## RAM usage during runing this block since iteratively plotting figures requires a lot of memory!

for p in list(set(clustering_predictions)):
    '''cick out the indexes of spectra belonging to each cluster'''
    idx = [i for i in range(len(pred)) if pred[i] == p]
    
    '''check how many different samples (or groups) contained in each cluster'''
    type = set(label[idx])
    
    '''output and plot the cluster meeting certain prerequisites'''
    if len(type) > 1:
    # if type == {1, 2, 3}:
    
        print(label[idx])
        print(group[idx])

        '''plot average spectrum'''
        # mean = np.mean(X[idx], axis=0)
        # ax.plot(raman_shift, mean, c='b', lw=2.0)
        # for t in idx:
        #     ax.plot(raman_shift, X[t], c='r', lw=0.8, alpha=0.3)

        '''plot spectra up to a certain number'''
        fig, ax = plt.subplots(3, 3, figsize=(9, 5), sharex='all', sharey='all')
        s, w = 0, 0
        for t in idx:
            ax[s, w].plot(raman_shift, X[t], c='r', lw=0.8)
            ax[s, w].set_title('str(label[t]))
            w += 1
            if w == 3:
                s += 1
                w = 0
            if s == 3:
                break

        # plt.show()
        # plt.close()

