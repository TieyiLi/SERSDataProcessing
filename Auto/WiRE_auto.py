import numpy as np
import pyautogui as agent
from time import sleep
from scipy.sparse import spdiags, linalg, csc_matrix
from scipy.fft import fft
from scipy.signal import savgol_filter
from renishawWiRE import WDFReader
import os

'''
BUTTON POSITIONS
Menu bar:
    Measurement: 280, 48
        Setup measurement: 280, 153
    Live video: 379, 48
        Save image: 379, 352
    Window: 752, 48
        close: 752, 104
        Window 1: 752, 338
        Window 2: 752, 363
        Window 3: TBD
Map measurement setup window:
    File: 245, 69
    File name: 313, 152
    Area setup: 606, 69
    X first: 570, 181
    X last: 663, 181
    Y first: 570, 207
    Y last: 663, 207
    OK: 455, 492
    Cancel: 564, 492
Sample Review:
    CCD camera turn on/off: 35, 727
    Light turn on/off: 75, 727
Video:
    upper left: 67, 158
    lower right: 815, 633
z_coordinate: 365, 118
x_coordinate: 189, 118
y_coordinate: 273, 118
GO to: 438, 118
'''

def check_button_positions():
    sleep(5)
    
    agent.click(752, 48)
    agent.moveTo(752, 338)
    sleep(1)
    agent.moveTo(752, 363)
    sleep(1)
    agent.click(752, 338)
    sleep(4)
    
    agent.click()
    agent.click(280, 48)
    agent.moveTo(280, 153)
    agent.click()
    sleep(2)
    agent.click(245, 69)
    agent.moveTo(313, 152)
    sleep(1)
    agent.click(606, 69)
    agent.moveTo(570, 181)
    sleep(1)
    agent.moveTo(663, 181)
    sleep(1)
    agent.moveTo(570, 207)
    sleep(1)
    agent.moveTo(663, 207)
    sleep(1)
    agent.click(564, 492)
    sleep(4)

    agent.moveTo(35, 727)
    sleep(1)
    agent.moveTo(75, 727)
    sleep(4)

    agent.moveTo(67, 158)
    sleep(1)
    agent.moveTo(815, 633)
    sleep(4)

    agent.moveTo(189, 118)
    sleep(1)
    agent.moveTo(273, 118)
    sleep(1)
    agent.moveTo(365, 118)
    sleep(1)
    agent.moveTo(438, 118)
    sleep(4)

    agent.click(280, 48)
    agent.click(280, 99)
    sleep(1)
    agent.moveTo(938, 643)
    sleep(1)
    agent.moveTo(1035, 451)
    agent.dragTo(1035, 619)
    agent.moveTo(655, 620)
    sleep(1)
    agent.click(1267, 720)

    
def auto_focus(z_series):
    scores = []
    for z in z_series:
        agent.click(365, 118)
        agent.hotkey('ctrl', 'a')
        agent.press('backspace')
        agent.write(str(z))
        agent.click(438, 118)
        sleep(1.0)
        image = agent.screenshot(region=(67, 158, 748, 475)).convert('L')
        im_array = np.asarray(image, dtype=np.int32)
        gy, gx = np.gradient(im_array)
        sharpness = np.mean(np.sqrt(gx**2 + gy**2))
        scores.append(sharpness)
    best_coarse_z = z_series[np.argmax(scores)]

    fine_z_series = np.arange(best_coarse_z - 2.0, best_coarse_z + 2.0, 0.5)
    fine_scores = []
    for z in fine_z_series:
        agent.click(365, 118)
        agent.hotkey('ctrl', 'a')
        agent.press('backspace')
        agent.write(str(z))
        agent.click(438, 118)
        sleep(1.0)
        image = agent.screenshot(region=(67, 158, 748, 475)).convert('L')
        im_array = np.asarray(image, dtype=np.int32)
        gy, gx = np.gradient(im_array)
        sharpness = np.mean(np.sqrt(gx**2 + gy**2))
        fine_scores.append(sharpness)
    best_z = fine_z_series[np.argmax(fine_scores)]
    
    agent.click(365, 118)
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(str(best_z))
    agent.click(438, 118)


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
    return np.array(ref_coordinates)


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
                if snr >= 8.0:
                    out.append([x_pos[progress], y_pos[progress]])
            progress += 1
    return out


def Initialize_area(init):
    agent.click(606, 69)
    agent.click(570, 181)
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(init)
    agent.click(663, 181)
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(init)
    agent.click(570, 207)
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(init)
    agent.click(663, 207)
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(init)


def Fill_area(x_low, x_high, y_low, y_high):
    agent.click(663, 181)
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(x_high)
    agent.click(570, 181)
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(x_low)
    agent.click(663, 207)
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(y_high)
    agent.click(570, 207)
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write(y_low)


def open_fine_map_template():
    agent.click(280, 48)
    agent.click(280, 99)
    sleep(1)
    agent.moveTo(1035, 451)
    agent.dragTo(1035, 619)
    agent.doubleClick(655, 620)
    sleep(1.5)


def open_coarse_map_template():
    agent.click(280, 48)
    agent.click(280, 99)
    sleep(1)
    agent.doubleClick(938, 643)
    sleep(1.5)


def restart_WiRE():
    '''Open templates'''
    '''Check camera and light status'''
    '''Active fine map'''
    '''Check if mapping can start without tunning laser or grating, if so the sleep time must be changed'''
    '''Rest for 5 mins'''
    return 0


def Run(save_folder_path, sample_name_str, area_cor, z_range, start_from=1):
    
    '''Parameter preset'''
    if not os.path.exists(save_folder_path):
        agent.alert(text='Saving path does not exist', title='File path error')
        raise ValueError('Folder path does not exist!')
    ref_cor = gen_coarse_coordinates(area_cor[0], area_cor[1])
    print(ref_cor)
    z_series = np.arange(z_range[0], z_range[1], 1)
    coarse_map_index = start_from
    
    '''Monitor software status'''
    fine_map_restart = 0
    app_restart = 0

    '''START'''
    for cor in ref_cor[start_from-1:]:

        '''Go to coarse map region and tune focus'''
        if coarse_map_index == 1:
            '''Open coarse map template'''
            open_coarse_map_template()

            '''Open fine map template'''
            open_fine_map_template()
        else:
            agent.click(189, 118)
            agent.hotkey('ctrl', 'a')
            agent.press('backspace')
            agent.write(str(cor[0]))
            agent.click(273, 118)
            agent.hotkey('ctrl', 'a')
            agent.press('backspace')
            agent.write(str(cor[1]))
            agent.click(438, 118)
            sleep(1.3)
            auto_focus(z_series)
            sleep(1.3)

        '''Go to next coarse map upper left position'''
        agent.click(752, 48)
        agent.click(752, 338)
        agent.click(280, 48)
        agent.click(280, 153)
        sleep(1.5)

        '''Set up coarse file path'''
        agent.click(245, 69)
        agent.click(313, 152)
        agent.hotkey('ctrl', 'a')
        agent.press('backspace')
        coarse_map_path = os.path.join(save_folder_path, 'map_' +
                                str(coarse_map_index) +
                                 '_' + sample_name_str) + '.wdf'
        agent.write(coarse_map_path)

        '''Set up coarse map area'''
        Initialize_area('0')
        Fill_area(str(cor[0]), str(cor[0] + 300.0),
                  str(cor[1]), str(cor[1] + 300.0))

        agent.click(455, 492)
        sleep(1.5)

        '''Running coarse map'''
        agent.press('f5')
        sleep(830)

        '''Extract positions for fine map'''
        positions_for_fine_map = extract_fine_map_positions(coarse_map_path)
        print('%d fine maps to be measured' % len(positions_for_fine_map))

        '''Runnning fine map'''
        if len(positions_for_fine_map) != 0:
            fine_map_index = 1

            '''Activate fine map window'''
            agent.click(752, 48)
            agent.click(752, 363)
            for [x_pos, y_pos] in positions_for_fine_map:

                '''Go to fine map setup'''
                agent.click(280, 48)
                agent.click(280, 153)
                sleep(1.5)
                
                '''Set up fine map path'''
                agent.click(245, 69)
                agent.click(313, 152)
                agent.hotkey('ctrl', 'a')
                agent.press('backspace')
                fine_map_path = os.path.join(save_folder_path, 'map_' +
                                               str(fine_map_index) + '@' +
                                               str(coarse_map_index) +
                                               '_' + sample_name_str) 
                agent.write(fine_map_path)
                
                '''Set up fine map area'''
                Initialize_area('-2')
                Fill_area(str(x_pos - 2.0), str(x_pos + 2.0),
                          str(y_pos - 2.0), str(y_pos + 2.0))
                
                agent.click(455, 492)
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
                if fine_map_restart == 30:
                    fine_map_restart = 0
                    agent.click(752, 48)
                    agent.click(752, 104)
                    sleep(2.5)
                    open_fine_map_template()
                    # agent.press('enter')
                    # sleep(1)
                    # agent.click(1133, 429)
                    # sleep(1)

        coarse_map_index += 1

        '''Restart WiRE'''
        app_restart += 1
        if app_restart == 5:
            restart_WiRE()
            app_restart = 0
            fine_map_restart = 0

        '''turn on light and camera'''
        agent.click(35, 727)
        sleep(3)
        
        

agent.FAILSAFE = True
agent.confirm('Start running', 'Confirm', buttons=['OK'])
