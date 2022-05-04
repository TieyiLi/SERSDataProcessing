import numpy as np
import pyautogui as agent
from time import sleep
from scipy.sparse import spdiags, linalg, csc_matrix
from scipy.fft import fft
from scipy.signal import savgol_filter
from renishawWiRE import WDFReader
import os


'''BUTTON POSITIONS'''

'''Menu bar'''
measurement = [280, 48]
setup_measurement = [280, 153]
open_template = [280, 99]
live_video = [379, 48]
set_origin = [367, 119]
window = [752, 48]
window_close = [752, 104]
window_1 = [752, 338]
window_2 = [752, 363]

'''Template'''
coarse_template = [938, 643]
scroll_bar = [1035, 451]
scroll_bar_down = [1035, 619]
fine_template = [655, 620]
template_cancel = [1267, 720]

'''Map measurement setup window'''
file = [245, 69]
file_name = [313, 152]
area_setup = [606, 69]
x_first = [570, 181]
x_last = [663, 181]
y_first = [570, 207]
y_last = [663, 207]
ok = [455, 492]
cancel = [564, 492]

'''Sample review'''
camera = [85, 857]
light = [123, 857]
view_upper_left = [67, 189]
view_lower_right = [814, 667]

'''Coordinates'''
x_coor = [189, 147]
y_coor = [273, 147]
z_coor = [365, 147]
go_to = [438, 147]

'''Others'''
open_WiRE = [176, 1057]
close_stage_error = [1169, 642]
task_bar_left = [1592, 1060]
task_bar_right = [1869, 1060, 2]
main_window = [1839, 555]


def check_button_positions():
    sleep(5)
    
    agent.click(window[0], window[1])
    agent.moveTo(window_1[0], window_1[1])
    sleep(1)
    agent.moveTo(window_2[0], window_2[1])
    sleep(1)
    agent.click(window_1[0], window_1[1])
    sleep(4)
    
    agent.click()
    agent.click(measurement[0], measurement[1])
    agent.moveTo(setup_measurement[0], setup_measurement[1])
    agent.click()
    sleep(2)
    agent.click(file[0], file[1])
    agent.moveTo(file_name[0], file_name[1])
    sleep(1)
    agent.click(area_setup[0], area_setup[1])
    agent.moveTo(x_first[0], x_first[1])
    sleep(1)
    agent.moveTo(x_last[0], x_last[1])
    sleep(1)
    agent.moveTo(y_first[0], y_first[1])
    sleep(1)
    agent.moveTo(y_last[0], y_last[1])
    sleep(1)
    agent.click(cancel[0], cancel[1])
    sleep(4)

    agent.moveTo(camera[0], camera[1])
    sleep(1)
    agent.moveTo(light[0], light[1])
    sleep(4)

    agent.moveTo(view_upper_left[0], view_upper_left[1])
    sleep(1)
    agent.moveTo(view_lower_right[0], view_lower_right[1])
    sleep(4)

    agent.moveTo(x_coor[0], x_coor[1])
    sleep(1)
    agent.moveTo(y_coor[0], y_coor[1])
    sleep(1)
    agent.moveTo(z_coor[0], z_coor[1])
    sleep(1)
    agent.moveTo(go_to[0], go_to[1])
    sleep(4)

    agent.click(measurement[0], measurement[1])
    agent.click(open_template[0], open_template[1])
    sleep(1)
    agent.moveTo(coarse_template[0], coarse_template[1])
    sleep(1)
    agent.moveTo(scroll_bar[0], scroll_bar[1])
    agent.dragTo(scroll_bar_down[0], scroll_bar_down[1])
    agent.moveTo(fine_template[0], fine_template[1])
    sleep(1)
    agent.click(template_cancel[0], template_cancel[1])

    
def auto_focus(z_series):
    scores = []
    for z in z_series:
        agent.click(z_coor[0], z_coor[1])
        agent.hotkey('ctrl', 'a')
        agent.press('backspace')
        agent.write(str(z))
        agent.click(go_to[0], go_to[1])
        sleep(1.0)
        image = agent.screenshot(region=(view_upper_left[0], view_upper_left[1], view_lower_right[0], view_lower_right[1])).convert('L')
        im_array = np.asarray(image, dtype=np.int32)
        gy, gx = np.gradient(im_array)
        sharpness = np.mean(np.sqrt(gx**2 + gy**2))
        scores.append(sharpness)
    best_coarse_z = z_series[np.argmax(scores)]

    fine_z_series = np.arange(best_coarse_z - 2.0, best_coarse_z + 2.0, 0.5)
    fine_scores = []
    for z in fine_z_series:
        agent.click(z_coor[0], z_coor[1])
        agent.hotkey('ctrl', 'a')
        agent.press('backspace')
        agent.write(str(z))
        agent.click(go_to[0], go_to[1])
        sleep(1.0)
        image = agent.screenshot(region=(view_upper_left[0], view_upper_left[1], view_lower_right[0], view_lower_right[1])).convert('L')
        im_array = np.asarray(image, dtype=np.int32)
        gy, gx = np.gradient(im_array)
        sharpness = np.mean(np.sqrt(gx**2 + gy**2))
        fine_scores.append(sharpness)
    best_z = fine_z_series[np.argmax(fine_scores)]
    
    agent.click(z_coor[0], z_coor[1])
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(str(best_z))
    agent.click(go_to[0], go_to[1])


def gen_coarse_coordinates(upper_left_coordinates, lower_right_coordinates):
    map_length = 310.0
    hori_length = lower_right_coordinates[0] - upper_left_coordinates[0]
    vert_length = lower_right_coordinates[1] - upper_left_coordinates[1]
    hori_No = hori_length // map_length - 1
    vert_No = vert_length // map_length - 1

    ref_coordinates = []
    h_idx, v_idx = 0, 0
    while v_idx <= vert_No:
        if h_idx == 0:
            while h_idx <= hori_No:
                ref_x = upper_left_coordinates[0] + h_idx * map_length
                ref_y = upper_left_coordinates[1] + v_idx * map_length
                ref_coordinates.append([ref_x, ref_y])
                h_idx += 1
                if h_idx == hori_No + 1:
                    v_idx += 1
        elif h_idx == hori_No + 1:
            while h_idx >= 1:
                h_idx -= 1
                ref_x = upper_left_coordinates[0] + h_idx * map_length
                ref_y = upper_left_coordinates[1] + v_idx * map_length
                ref_coordinates.append([ref_x, ref_y])
                if h_idx == 0:
                    v_idx += 1
        else:
            raise ValueError('Coordinates wrong!')
    return ref_coordinates


def baseline_sub(x, lam=1e4, p=0.005, niter=10):
    L = len(x)
    D = csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    baseline = 0
    for i in range(niter):
        W = spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        baseline = linalg.spsolve(Z, w * x)
        w = p * (x > baseline) + (1 - p) * (x < baseline)
    return x - baseline


def extract_fine_map_positions(coarse_map_path):
    reader = WDFReader(coarse_map_path)
    wavenumber, collections = reader.xdata, reader.spectra
    x_pos, y_pos = reader.xpos, reader.ypos
    progress = 0
    out = []
    for i in range(collections.shape[0]):
        for j in range(collections.shape[1]):
            spectrum = collections[i, j]
            base_sub = baseline_sub(spectrum)
            norm_spectrum = (base_sub - min(base_sub)) / (max(base_sub) - min(base_sub))
            score = np.mean(np.abs(np.real(fft(norm_spectrum)[16:32])))
            if score >= 4.9:
                smooth = savgol_filter(base_sub, 7, 1)
                N = np.mean(abs(smooth - base_sub))
                snr = max(smooth) / N
                if snr >= 4.0:
                    out.append([x_pos[progress], y_pos[progress]])
            progress += 1
    return out


def Initialize_area(init):
    agent.click(area_setup[0], area_setup[1])
    agent.click(x_first[0], x_first[1])
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(init)
    agent.click(x_last[0], x_last[1])
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(init)
    agent.click(y_first[0], y_first[1])
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(init)
    agent.click(y_last[0], y_last[1])
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(init)


def Fill_area(x_low, x_high, y_low, y_high):
    agent.click(x_first[0], x_first[1])
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(x_high)
    agent.click(x_last[0], x_last[1])
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(x_low)
    agent.click(y_first[0], y_first[1])
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(y_high)
    agent.click(y_last[0], y_last[1])
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(y_low)


def open_fine_map_template():
    agent.click(measurement[0], measurement[1])
    agent.click(open_template[0], open_template[1])
    sleep(1)
    agent.moveTo(scroll_bar[0], scroll_bar[1])
    agent.dragTo(scroll_bar_down[0], scroll_bar_down[1])
    agent.doubleClick(fine_template[0], fine_template[1])
    sleep(1.5)


def open_coarse_map_template():
    agent.click(measurement[0], measurement[1])
    agent.click(open_template[0], open_template[1])
    sleep(1)
    agent.doubleClick(coarse_template[0], coarse_template[1])
    sleep(1.5)


def restart_WiRE(cor):

    agent.click(x_coor[0], x_coor[1])
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(str(cor[0]))
    agent.click(y_coor[0], y_coor[1])
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(str(cor[1]))
    agent.click(go_to[0], go_to[1])
    sleep(1.3)
    
    os.system('wmic process where name="WiREInterface.exe" delete')
    os.system('wmic process where name="WiREQueue.exe" delete')
    sleep(360)
    agent.click(open_WiRE[0], open_WiRE[1])
    sleep(20)
    agent.click(close_stage_error[0], close_stage_error[1])
    sleep(90)

    '''Set origin'''
    agent.click(live_video[0], live_video[1])
    agent.click(set_origin[0], set_origin[1])
    sleep(0.5)

    agent.click(x_coor[0], x_coor[1])
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(str(-cor[0]))
    agent.click(y_coor[0], y_coor[1])
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(str(-cor[1]))
    agent.click(go_to[0], go_to[1])
    sleep(1.3)

    '''Set origin'''
    agent.click(live_video[0], live_video[1])
    agent.click(set_origin[0], set_origin[1])
    sleep(0.5)

    open_coarse_map_template()
    open_fine_map_template()
    agent.moveTo(task_bar_left[0], task_bar_left[1])
    agent.moveTo(task_bar_right[0], task_bar_right[1], 2)
    agent.moveTo(main_window[0], main_window[1])
    sleep(0.5)
    

def Run(save_folder_path, sample_name_str, area_cor, coarse_id_from=1):
    
    '''Parameter preset'''
    if not os.path.exists(save_folder_path):
        agent.alert(text='Saving path does not exist', title='File path error')
        raise FileNotFoundError('Folder path does not exist!')
    ref_cor = gen_coarse_coordinates(area_cor[0], area_cor[1])
    print(ref_cor)
    z_series = np.arange(-15, 15, 1)
    
    '''Monitor software status'''
    fine_map_restart = 0
    app_restart = 0

    '''START'''
    for cor in ref_cor:

        '''Move to coarse map upper left position'''
        if cor == ref_cor[0]:
            pass
        else:
            agent.click(x_coor[0], x_coor[1])
            agent.hotkey('ctrl', 'a')
            agent.press('backspace')
            agent.write(str(cor[0]))
            agent.click(y_coor[0], y_coor[1])
            agent.hotkey('ctrl', 'a')
            agent.press('backspace')
            agent.write(str(cor[1]))
            agent.click(go_to[0], go_to[1])
            sleep(1.3)
            auto_focus(z_series)
            sleep(1.3)

        '''Open measurement setup'''
        agent.click(window[0], window[1])
        agent.click(window_1[0], window_1[1])
        agent.click(measurement[0], measurement[1])
        agent.click(setup_measurement[0], setup_measurement[1])
        sleep(1.5)

        '''Set up coarse file path'''
        agent.click(file[0], file[1])
        agent.click(file_name[0], file_name[1])
        agent.hotkey('ctrl', 'a')
        agent.press('backspace')
        coarse_map_path = os.path.join(save_folder_path, 'map_' +
                                str(coarse_id_from) +
                                 '_' + sample_name_str) + '.wdf'
        agent.write(coarse_map_path)

        '''Set up coarse map area'''
        Initialize_area('0')
        Fill_area(str(cor[0]), str(cor[0] + 300.0),
                  str(cor[1]), str(cor[1] + 300.0))

        agent.click(ok[0], ok[1])
        sleep(1.5)

        '''Running coarse map'''
        agent.press('f5')
        if app_restart == 0:
            sleep(890)
        else:
            sleep(860)

        '''Extract positions for fine map'''
        positions_for_fine_map = extract_fine_map_positions(coarse_map_path)
        print('%d fine maps to be measured' % len(positions_for_fine_map))

        '''Runnning fine map'''
        if len(positions_for_fine_map) != 0:
            fine_map_index = 1

            '''Activate fine map window'''
            agent.click(window[0], window[1])
            agent.click(window_2[0], window_2[1])
            for [x_pos, y_pos] in positions_for_fine_map:

                '''Go to fine map setup'''
                agent.click(measurement[0], measurement[1])
                agent.click(setup_measurement[0], setup_measurement[1])
                sleep(1.5)
                
                '''Set up fine map path'''
                agent.click(file[0], file[1])
                agent.click(file_name[0], file_name[1])
                agent.hotkey('ctrl', 'a')
                agent.press('backspace')
                fine_map_path = os.path.join(save_folder_path, 'map_' +
                                               str(fine_map_index) + '@' +
                                               str(coarse_id_from) +
                                               '_' + sample_name_str) 
                agent.write(fine_map_path)
                
                '''Set up fine map area'''
                Initialize_area('-2')
                Fill_area(str(x_pos - 2.0), str(x_pos + 2.0),
                          str(y_pos - 2.0), str(y_pos + 2.0))
                
                agent.click(ok[0], ok[1])
                sleep(1.2)

                '''Running fine map'''
                agent.press('f5')
                if fine_map_index == 1:
                    sleep(28)
                else:
                    sleep(25)

                fine_map_index += 1
                
                '''Restart fine map to release RAM'''
                fine_map_restart += 1
                if fine_map_restart == 25:
                    fine_map_restart = 0
                    agent.click(window[0], window[1])
                    agent.click(window_close[0], window_close[1])
                    sleep(2.5)
                    open_fine_map_template()

        coarse_id_from += 1

        '''Restart WiRE'''
        app_restart += 1
        if app_restart == 4:
            restart_WiRE(cor)
            app_restart = 0
            fine_map_restart = 0

        '''turn on light and camera'''
        agent.click(camera[0], camera[1])
        sleep(3)
        

agent.FAILSAFE = True
agent.confirm('Start running ?', 'Confirm', buttons=['OK'])
