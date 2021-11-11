"""
===================================
Prepare data for following analysis
===================================
This class serves as the bridge connnecting spectra colletions and numerical analysis 
including classification, clustering and so on.
"""


import numpy as np
import re, shutil, os
From Utils import Utils


class PrepareData(Utils):

    def ranking_selection(self, data_from, data_to, n_each_map, thr=22, copy=False):
        """
        This function picks the spectra that are qualified for the following
        analysis.

        Requirements
        ----------
        The sub-folder structure of 'data_from' directory must be 'samples\mappings\spectra.txt'.

        Parameters
        ----------
        data_from: path of directory containing samples stored by spectra (.txt) files that
                   are exported by 'export_wdf' function
        data_to: path of directory for saving qualified spectra with the same sub-folder
                 structure as 'data_from' directory
        n_each_map: number of spectra to be picked from each map
        thr: minimal quality of spectrum in order to be picked, must be 22 or higher currently
        copy: Saving qualified spectra or not

        Returns
        -------
        None
        """
        selected = 0
        pattern = re.compile(r'\\')

        # Iterate all spectra.txt
        for datadir, mapdir, txtlist in os.walk(data_from):
            data_scores = []
            data_path_list = []
            for txtname in txtlist:

                # Identify only .txt files and ignore other file types
                if txtname[-3:] == 'txt':
                    txtpath = os.path.join(datadir, txtname)
                    data_with_raman_shift = np.loadtxt(txtpath)

                    # Keep only 2nd dimension representing intensities
                    data = data_with_raman_shift[:, 1]
                    snr, sig = self.estimate_snr(data)

                    # Ignore spectra.txt containing intensities saturation or
                    # cosmic ray
                    if np.sum(data == 0) <= 5 and self.check_single_spike(sig):
                        data_scores.append(snr)
                        data_path_list.append(txtpath)

            if data_scores:

                # Sort the stored spectra by quality
                sorted_idx = np.argsort(data_scores)
                for best_idx in range(n_each_map):

                    # Pick spectra with restriction of maximal number of spectra to be picked
                    # as well as the minimal required quality
                    if abs(1 + best_idx) <= len(data_path_list) and \
                            data_scores[sorted_idx[-1 - best_idx]] >= thr:

                        raw_txt_path = data_path_list[sorted_idx[-1 - best_idx]]
                        splited = pattern.split(raw_txt_path)
                        sample_path = splited[-3]
                        map_path = splited[-2]
                        txt_name = splited[-1]
                        new_folder_path = os.path.join(data_to, sample_path, map_path)
                        print(raw_txt_path)
                        selected += 1
                        if copy:
                            if not os.path.exists(new_folder_path):
                                os.makedirs(new_folder_path)
                            new_txt_path = os.path.join(new_folder_path, txt_name)
                            shutil.copyfile(raw_txt_path, new_txt_path)
                    else:
                        break

        print('%d are selected!' % selected)

    def gen_label_dict(self, data_path):
        """
        Convenient function for generating 'label_dict' for 'read_data'.
        """
        
        start = 1
        for dir in os.listdir(data_path):
            dict[dir] = start
            start += 1
        print(dict)
        return dict

    def read_data(self, data_path, label_dict, group_dict, interp=False):

        """
        This function stores the qualified spectra.txt folder to a customized
        dictionary that is to be saved as (.npy), which is the key to all of 
        the following analysis.

        Requirements
        ----------
        Must be only used for storing qualified spectra.txt instead of original
        spectra.txt.

        Parameters
        ----------
        data_path: path of samples folder with qualified spectra.txt, must have the sub-folder
                   structure of 'samples\mappings\qualified spectra.txt'
        label_dict: dictionary for labeling each sample,
                    Must in the format of {'exact directory name of each sample': integer} such
                    as {'cnt_1': 1, 'cnt_2': 2, ...}
        group_dict: dictionary for grouping samples according to higher-level properties (e.g.
                    patient versus healthy control), typically using 0 for normal and
                    1 for diseased and -1 for unknown
                    Must in the format of {integer: list of sample labels or array of sample
                    labels} such as {0: [1, 2, 3, 4, 5], 1: [6, 7, 8, 9, 10]} or {0: np.arrange(
                    1, 6), 1: np.arrange(6, 11)}
        interp: interpolating spectra or not, default 'False', when spectra are collected from
                different Raman spectroscopy, this must be set to 'True'
                
        Dictionary keys meaning
        ----------
        data_matrix: 2D numpy array with the shape of (number of spectra, number of Raman shifts) 
                     including all the spectra collection in a single study
        sample_name: sample name of each spectra stored in a list
        map_index: mapping name of each spectra stored in a list
        label: 1D numpy array labeling each spectra with the sample number given by 'label_dict'
        group: 1D numpy array labeling each spectra with higher-level group number given by 
               'group_dict'
        raman_shift: 1D numpy array storing the common Raman shift of all spectra
        file_path: original (.txt) file path of each spectra
        label_dict: 'label_dict' given in parameters
        group_dict: 'group_dict' given in parameters
        
        Returns
        -------
        Dictionary of spectra and corresponding properties
        """

        sample_name, map_index, txt_path_list, data_matrix, label, group = [], [], [], [], [], []
        data_dict = {}

        pattern = re.compile(r'\\')

        for datadir, mapdir, txtlist in os.walk(data_path):
            for txtname in txtlist:
                if txtname[-3:] == 'txt':
                    file_name_splited = pattern.split(datadir)
                    mapindex = file_name_splited[-1]
                    samplename = file_name_splited[-2]
                    map_index.append(os.path.join(samplename, mapindex))
                    sample_name.append(samplename)
                    label.append(label_dict[samplename])
                    for key in group_dict.keys():
                        if label_dict[samplename] in group_dict[key]:
                            group.append(key)
                    txt_path = os.path.join(datadir, txtname)
                    txt_path_list.append(txt_path)

        if interp:
            raman_shift = np.arange(564, 1681, 1)
            for txt_path in txt_path_list:
                data_line = np.loadtxt(txt_path)
                intensity = data_line[:, 1]
                raman_shift_ = data_line[:, 0]
                interp_data = np.interp(raman_shift, raman_shift_[::-1], intensity[::-1])
                data_matrix.append(interp_data)
        else:
            for txt_path in txt_path_list:
                data_line = np.loadtxt(txt_path)
                data_matrix.append(data_line[:, 1].reshape(1, -1)[0, :])

            raman_shift = np.loadtxt(txt_path_list[0])[:, 0]

        print(np.array(data_matrix).shape)
        data_dict['data_matrix'] = np.array(data_matrix)
        data_dict['sample_name'] = sample_name
        data_dict['map_index'] = map_index
        data_dict['label'] = np.array(label)
        data_dict['group'] = np.array(group)
        data_dict['raman_shift'] = raman_shift
        data_dict['file_path'] = txt_path_list
        data_dict['label_dict'] = label_dict
        data_dict['group_dict'] = group_dict

        return data_dict

    def save(self, data_dict, pro_path_dir, pro_file_name, base_sub=True, smoothing=False, norm=True):
        """
        Saving data dictionary as (.npy) with defined preprocessing steps.
        
        Parameters
        ----------
        data_dict: data dictionary
        pro_path_dir: path of directory for saving 
        pro_file_name: name of (.npy) file
        base_sub: subtract baseline, default 'True'
        smoothing: noise reduction, default 'False'
        norm: normalization, default 'True'

        Returns
        -------
        None
        """

        X = np.copy(data_dict['data_matrix'])
        if base_sub:
            X = self.baseline_sub(X)
        if smoothing:
            X = self.smoothing(X)
        if norm:
            X = self.normalize(X)
        data_dict['data_matrix'] = X
        
        pro_file_path = os.path.join(pro_path_dir, pro_file_name)
        np.save(pro_file_path, data_dict)
        
"""
Example usage:
1) Picking spectra
>>> source_path = r'D:\Raman Data\Jun Liu\CoH serum sample\txt_useful'
>>> export_path = r'D:\Raman Data\Jun Liu\CoH serum sample\txt_useful_5each'
>>> pd = PrepareData(source_path, export_path)
>>> pd.ranking_selection(n_each_map=5, thr=22, copy=True) # comment this line after finishing spectra selection

2) Create and save data dictionary
>>> data_path = r'D:\Raman Data\Jun Liu\CoH serum sample\txt_useful_5each'
>>> label_dict = pd.gen_label_dict(data_path) # use only per time
    or label_dict = {'ACE2': 1, 'Cov2_B117': 2, 'Cov2_Delta': 3, 'Cov2_WA': 4}
    or label_dict = {'v1': 1, 'v2': 2, 'v3': 3, 'v4': 4, 'v7': 7, 'v8': 8, 'v5': 5, 'v6': 6, 'v9': 9, 'v10': 10}
    or label_dict = {'cnt_1': 1, 'cov2_1': 11, 'cov2_2': 12, 'cov2_3': 13, 'cov2_4': 14, 'cnt_2': 2,
                     'cnt_3': 3, 'cov2_5': 15, 'cnt_4': 4, 'cov2_6': 16,  'cnt_5': 5, 'cov2_7': 17,
                     'cov2_8': 18, 'cov2_9': 19, 'cnt_7': 7, 'cnt_6': 6, 'cnt_8': 8, 'cov2_10': 20,
                     'cnt_9': 9, 'cnt_10': 10}
    or label_dict = {'colon1': 6, 'control5': 5, 'control4': 4, 'control3': 3, 'control2': 2,
                     'control1': 1, 'colon10': 10, 'colon8': 9, 'colon5': 8, 'colon2': 7}

>>> group_dict = {-1: [1, 2, 3, 4, 5], 1: [6, 7, 8, 9, 10]}
>>> dir = r'D:\Tieyi Li\datanpy\Jun'
>>> file_name = 'data11_8_21_CoHSerumSamples_Colon&Control_5Each_Sub' # please following this naming standard

>>> data_dict = pd.read_data(data_path, label_dict, group_dict, interp=False) # comment this line after finishing to save running time
>>> pd.save(data_dict, dir, file_name, base_sub=True, smoothing=False, norm=True) # also comment this line to prevent (.npy) file overwritten
"""
