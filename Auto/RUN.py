from WiRE_AUTO import check_button_positions, Run
import pyautogui as agent

#================================AUTO MEASUREMENT==========================#
#----------------------DO check button positions before running!!!---------#
##check_button_positions()
#---------------------------------Start running----------------------------#
#REQIUREMENTS:
# 1) Coarse mapping and fine mapping templetes are open;
# 2) Staying at origion (0, 0) with good focus;
# 3) Laser and grating are ready to use without laser or grating switching;
# 4) Crosshairs and scalebar in Video window are recommended to be closed.
# 5) Turn off administrator confirmation notification

save_folder_path = r'C:\Users\Tieyi Li\Auto run test v2'
sample_name_str = 'Cnt5Exo'
area_cor = [[0, 0], [1990, 1785]]
z_range = [-20, 15]

Run(save_folder_path, sample_name_str, area_cor, z_range)
