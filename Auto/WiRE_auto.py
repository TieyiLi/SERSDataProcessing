from pywinauto import application
from time import sleep
import pyautogui as agent

def calculate_sharpness(im):
    im_array = np.asarray(im, dtype=np.int32)
    gy, gx = np.gradient(im_array)
    sharpness = np.average(np.sqrt(gx**2 + gy**2))
    return sharpness

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
