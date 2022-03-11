from pywinauto import application
from time import sleep
import pyautogui as agent
from scipy.sparse import spdiags, linalg, csc_matrix
from scipy.signal import savgol_filter
from scipy.fft import fft
from renishawWiRE import WDFReader
import numpy as np

def auto_focus():
    z_series = np.arange(-15, 15, 0.5)
    scores = []
    for z in z_series:
        image = agent.screenshot().convert('L')
        im_array = np.asarray(image, dtype=np.int32)
        gy, gx = np.gradient(im_array)
        sharpness = np.mean(np.sqrt(gx**2 + gy**2))
        scores.append(sharpness)
    best_z = z_series[np.argmax(scores)]
    return best_z


def gen_coarse_coordinates(upper_left_coordinates, lower_right_coordinates):
    map_length = 305.0
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


def baseline_sub(self, x, lam=1e4, p=0.005, niter=10):
    '''lam usually from 100-10^9, p from 0.000001 to 0.1'''
    '''used 300, 0.01'''
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


def metrics_for_coarse_map(coarse_map_path):
    reader = WDFReader(coarse_map_path)
    wavenumber, collections = reader.xdata, reader.spectra
    x_pos, y_pos = reader.xpos, reader.ypos
    progress = 0
    out = []
    for i in range(collections.shape[0]):
        for j in range(collections.shape[1]):
            spectrum = collections[i, j]
            base_sub = baseline_sub(spectrum)
            norm_spectrum = (spectrum - min(spectrum)) / (max(spectrum) - min(spectrum))
            score = np.mean(np.abs(np.real(fft(norm_spectrum)[16:32])))
            if score >= 4.9:
                smooth = savgol_filter(base_sub, 7, 1)
                N = np.mean(abs(smooth - base_sub))
                snr = max(smooth) / N
                if snr >= 8.0:
                    out.append([x_pos[progress], y_pos[progress]])
            progress += 1
    return out


##while True:
##    x, y = agent.position()
##    positionStr = 'X: ' + str(x).rjust(4) + ' Y: ' + str(y).rjust(4)
##    print(positionStr)
##    sleep(4)

'''
BUTTON POSITIONS

Menu bar:
    Measurement: 280, 48
        Setup measurement: 280, 153
    Live video: 379, 48
        Save image: 379, 352
    Window: 752, 48
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
    CCD camera turn on/off: 28, 764
    Light turn on/off: 68, 764
'''

def test_menubar():
    agent.click(606, 69)
    agent.click(570, 181)
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write('380')
    agent.click(663, 181)
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write('389')
    agent.click(570, 207)
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write('380')
    agent.click(663, 207)
    agent.hotkey('ctrl', 'a')
    agent.press('backspace')
    agent.write('389')
    agent.click(564, 492)

def test_sample_review_control():
    agent.click(68, 764, clicks=2, interval=2)


# Open setup measurement window
agent.FAILSAFE = True
agent.click(280, 48)
agent.click(280, 153)
sleep(1.5)
agent.click(245, 69)
agent.click(313, 153)
agent.hotkey('ctrl', 'a')
agent.press('backspace')

# Change file name and setup area
agent.click(606, 69)
agent.click(570, 181)
agent.hotkey('ctrl', 'a')
agent.press('backspace')
agent.write('380')
agent.click(663, 181)
agent.hotkey('ctrl', 'a')
agent.press('backspace')
agent.write('389')
agent.click(570, 207)
agent.hotkey('ctrl', 'a')
agent.press('backspace')
agent.write('380')
agent.click(663, 207)
agent.hotkey('ctrl', 'a')
agent.press('backspace')
agent.write('389')
agent.click(564, 492)

# Turn on camera and light
agent.click(68, 764, clicks=2, interval=2)
