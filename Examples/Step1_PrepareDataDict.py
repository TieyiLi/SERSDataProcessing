from Tools.PrepareData import PrepareData
from Tools.ExportWDF import export_wdf


##===================================================Export WDF to TXT===========================================================##
## only applicable to coarse+fine mapping
source_dir_path = 'E:\Raman data\WDF data\12_21_20_ColonExosome'
export_dir_path = 'E:\Raman data\TXT data\12_21_20_ColonExosome'

## Do comment the following line when idle
# export_wdf(source_folder_path=source_dir_path, export_folder_path=export_dir_path)

##=================================================Pick Qualified Spectra========================================================##

source_path = r'D:\Raman Data\Jun Liu\CoH serum sample\txt_useful'
export_path = r'D:\Raman Data\Jun Liu\CoH serum sample\txt_useful_5each'
pd = PrepareData(source_path, export_path)

## Do comment the following line when idle
# pd.ranking_selection(n_each_map=5, thr=22, copy=True)

##=============================================Create Data Dictionary and Save===================================================##


data_path = r'D:\Raman Data\Jun Liu\CoH serum sample\txt_useful_5each'

## use only label_dict per time
label_dict = pd.gen_label_dict(data_path) 
# label_dict = {'ACE2': 1, 'Cov2_B117': 2, 'Cov2_Delta': 3, 'Cov2_WA': 4}
# label_dict = {'v1': 1, 'v2': 2, 'v3': 3, 'v4': 4, 'v7': 7, 'v8': 8, 'v5': 5, 'v6': 6, 'v9': 9, 'v10': 10}
# label_dict = {'cnt_1': 1, 'cov2_1': 11, 'cov2_2': 12, 'cov2_3': 13, 'cov2_4': 14, 'cnt_2': 2,
#               'cnt_3': 3, 'cov2_5': 15, 'cnt_4': 4, 'cov2_6': 16,  'cnt_5': 5, 'cov2_7': 17,
#               'cov2_8': 18, 'cov2_9': 19, 'cnt_7': 7, 'cnt_6': 6, 'cnt_8': 8, 'cov2_10': 20,
#               'cnt_9': 9, 'cnt_10': 10}
# label_dict = {'colon1': 6, 'control5': 5, 'control4': 4, 'control3': 3, 'control2': 2,
                     'control1': 1, 'colon10': 10, 'colon8': 9, 'colon5': 8, 'colon2': 7}
                        
group_dict = {-1: [1, 2, 3, 4, 5], 1: [6, 7, 8, 9, 10]}
dir = r'D:\Tieyi Li\datanpy\Jun'
file_name = 'data11_8_21_CoHSerumSamples_Colon&Control_5Each_Sub' # please following this naming standard

## Do comment the following 2 lines when idle
# data_dict = pd.read_data(data_path, label_dict, group_dict, interp=False)
# pd.save(data_dict, dir, file_name, base_sub=True, smoothing=False, norm=True)
