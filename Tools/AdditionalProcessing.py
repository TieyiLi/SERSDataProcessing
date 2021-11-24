"""
===============================
Additional processing functions 
===============================
This class contains additional useful functions such as relabeling, summarizing 
test results and so on.
"""

import numpy as np
from Tools.Utils import Utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


class AdditionalProcessing(Utils):
    
    def clustering_summary(self, cluster_pred):
        
        cluster_size_summary = []
        pred_set = set(cluster_pred)
        for s in pred_set:
            idx_s = [i for i in range(len(cluster_pred)) if cluster_pred[i] == s]
            cluster_size_summary.append(len(idx_s))
        return cluster_size_summary


    def relabel_by_purity(self, cluster_pred, ori_label):
        """
        This function relabels the data in the way that only cluster exlusively contains diseased
        samples is labeled as diseased.
      
        Parameters
        ----------
        cluser_pred: clustering predictions with the shape of (number of spectra, )
        ori_label: original label to be relabeled, usually using the one labeling normal or diseased
        
        Returns
        -------
        New label with the same shape
        """
        
        new_label = np.zeros_like(ori_label)
        total_cluster_num = 0
        total_interest_num = 0
        L = len(ori_label)
        pred_set = set(cluster_pred)
        for s in pred_set:
            idx_s = [i for i in range(L) if cluster_pred[i] == s]
            if np.all(ori_label[idx_s] == 1):
                new_label[idx_s] = 1
                total_cluster_num += 1
                total_interest_num += len(idx_s)

        print('%d/%d specific clusters' % (total_cluster_num, len(pred_set)))
        print('%d/%d interest instances' % (total_interest_num, L))

        return new_label
    
    
    def split_mixed_maps(self, relabeled_group, ori_map_index):
        """
        This function splits the maps which contains both negative and positive spectra
        after relabeling. This function only applied to binary classification.
      
        Parameters
        ----------
        relabeled_group: relabeled group (data dictionary 'group' component) with the shape of (number of spectra, )
        ori_map_index: original map index (data dictionary 'map_index' component)
        
        Returns
        -------
        New map index list
        """
        new_map_index = ori_map_index.copy()
        L = len(relabeled_group)
        for type in set(ori_map_index):
            idx = [i for i in range(L) if ori_map_index[i] == type]
            sub_group = relabeled_group[idx]
            if set(sub_group) == {0, 1}:
                idx_pos = [idx[j] for j in range(len(idx)) if sub_group[j] == 1]
                for p in idx_pos:
                    new_map_index[p] = ori_map_index[p] + '_p'
        return new_map_index


    def relabel_by_ratio(self, cluster_pred, ori_label, percentage=0.5):
        """
        This function relabels the data in the way that cluster contains diseased spectra fraction 
        more than a percentage threshold is labeled as diseased.
      
        Parameters
        ----------
        cluser_pred: clustering predictions with the shape of (number of spectra, )
        ori_label: original label to be relabeled, usually using the one labeling normal or diseased
        percentage: threshold percentage of diseased spectra fraction
        
        Returns
        -------
        New label with the same shape
        """
        
        if set(ori_label) != {0, 1}:
            raise ValueError('Relabel can not be applied to multi-labels!')

        if not isinstance(cluster_pred, np.ndarray):
            cluster_pred = np.array(cluster_pred)
        if not isinstance(ori_label, np.ndarray):
            ori_label = np.array(ori_label)

        new_label = np.empty(shape=cluster_pred.shape)
        cluster_set = set(cluster_pred)
        for set_i in cluster_set:
            idx = [i for i in range(len(cluster_pred)) if cluster_pred[i] == set_i]

            group_label_i = ori_label[idx]
            types = list(set(group_label_i))
            if len(types) == 1:
                new_label[idx] = types[0]
            else:
                num_tot = len(group_label_i)
                num_p = len(group_label_i[group_label_i == 1])
                if num_p >= percentage * num_tot:
                    new_label[idx] = 1
                else:
                    new_label[idx] = 0

        return new_label

    def input_data(self, data_path, output='content'):
        """
        Read data dictionary which is ready to be analyzed.
        """
        data_dict = np.load(data_path, allow_pickle=True).item()
        if output == 'content':
            X = data_dict['data_matrix']
            label = data_dict['label']
            group = data_dict['group']
            sample_name = data_dict['sample_name']
            raman_shift = data_dict['raman_shift']
            map_index = data_dict['map_index']
            return X, label, group, sample_name, raman_shift, map_index
        elif output == 'dict':
            return data_dict
        else:
            raise ValueError('Option is not defined!')

    def test_summary_spec_base(self, test_sample_label, y_pred):
        """
        This function generates a summary of test with counting the number of spectra instead of 
        mappings (vesicles or particles).
      
        Parameters
        ----------
        test_sample_label: labels labeling each sample in the test dataset
        y_pred: predictions of a model running on the test dataset
        
        Returns
        -------
        None
        """
        individual = set(test_sample_label)
        pred_type = set(y_pred)
        for i in individual:
            print('Individual %d:' % i)
            idx = [j for j in range(len(test_sample_label)) if test_sample_label[j] == i]
            pred_i = y_pred[idx]
            for t in pred_type:
                num = len(pred_i[pred_i == t])
                print('    predicted as %d: %d' % (t, num), end='')
                print('(%.3f)' % (num / len(pred_i)))

    def test_summary_map_base(self, test_sample_label, map_index, y_pred):
        """
        This function generates a summary of test with counting the number of mappings (vesicles
        or particles) instead of spectra.
      
        Parameters
        ----------
        test_sample_label: labels labeling each sample in the test dataset
        map_index: mapping name of each spectra stored in the data dictionary
        y_pred: predictions of a model running on the test dataset
        
        Returns
        -------
        None
        """
        
        sample_type = set(test_sample_label)
        s = 0
        for i in sample_type:
            content_summary = []
            print('Individual %d:' % i)
            sample_idx = [j for j in range(len(test_sample_label)) if test_sample_label[j] == i]
            pred_i = y_pred[sample_idx]

            map_for_sample = [map_index[s] for s in sample_idx]
            map_type = set(map_for_sample)
            for m in map_type:
                map_idx = [u for u in range(len(map_for_sample)) if map_for_sample[u] == m]
                pred_m = pred_i[map_idx]
                most_common_ele = np.argmax(np.bincount(pred_m))
                content_summary.append(most_common_ele)

            w = 0
            for n in set(content_summary):
                len_n = len([v for v in range(len(content_summary)) if content_summary[v] == n])
                w += 1
                print('    Prediction of %d: %d' % (int(n), len_n), end='')
                print('(%.3f)' % (len_n / len(content_summary)))
            s += 1
            
  
  def manual_roc(y_true, y_score):

    fpr, tpr, thr = roc_curve(y_true, y_score, pos_label=1)
    print(tpr)
    print(thr)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7))
    ax.plot(fpr, tpr, 'deepskyblue', lw=2.0)
    ax.set_xlabel('1 - Specificity', fontsize=13, fontstyle='italic')
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_ylabel('Sensitivity', c='deepskyblue', fontsize=13, fontstyle='italic')
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.plot([0, 1], [0, 1], 'deepskyblue', linestyle='-.', lw=1.0)

    ax2 = ax.twinx()
    ax2.plot(fpr, thr, 'orange', marker='o', lw=1.6, markersize=3)
    ax2.set_ylabel('Threshold', c='orange', fontsize=13, fontstyle='italic')

    ax.set_facecolor('#EAEAF2')
    ax_yticks = ax.get_yticks()
    ax.grid(True, color='white')
    ax2.set_yticks(np.linspace(np.min(thr), np.max(thr), len(ax_yticks)))

    ax.spines['bottom'].set_color('white')
    ax2.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax2.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax2.spines['right'].set_color('white')

    ax.tick_params(axis='y', colors='deepskyblue')
    ax2.tick_params(axis='y', colors='orange')

    plt.show()
