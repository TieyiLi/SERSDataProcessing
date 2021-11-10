from Tools.Utils import PrepareData

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
