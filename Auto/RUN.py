from WiRE_AUTO import check_button_positions, Run
import pyautogui as agent

#================================AUTO MEASUREMENT==========================#
#----------------------DO check button positions before running!!!---------#
#check_button_positions()
#---------------------------------Start running----------------------------#
#REQIUREMENTS:
# 1) Coarse mapping (1st) and fine mapping (2nd) templetes are open;
# 2) Staying at origion (0, 0) with good focus;
# 3) Laser and grating are ready to use without laser or grating switching;
# 4) Crosshairs and scalebar in Video window are recommended to be closed.
# 5) Turn off administrator confirmation notification.

save_folder_path = r'C:\Users\Siddharth Srivastava\eye_exo\round3\09_01'
sample_name_str = '09_01'
area_cor = [[0, 0], [2500, 2200]]

Run(save_folder_path, sample_name_str, area_cor, coarse_id_from=1)
