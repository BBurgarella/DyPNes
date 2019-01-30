# DyPNes

DypNes is a bot made to control nes games. it was created by Boris Burgarella and is released under the General Public Licence

Dependencies (usually included in Abaqus and / or default python installations):
- Pywin32
- Pyautogui
- mss
- Keras 
- matplotlib
- numpy
- pickle

If you want to train you own version of the bot, you can edit the parameters in the config.py file, pre-trained models will be uploaded in close future. 

The current version of the model works with a 1 input tensor, consisting in four concatenated successive frames and the output is divided in two vectors: a 4x1 for the direction and a 2x1 of the A and B buttons. The Neural network consists for now in a first part made of convolution layers with LeakyReLU activations, the output of this first part is then fed to a dense network to predict the direction. Finally, the output of the convolution layers and the direction is then fed to a third (dense) network to predict if A or B should be pushed

The default keybinds are mapped for an azerty keyboard as show on the ascii bellow. it works as intended on the emulator "MESEN" and has not been tested on other emulators.

                             | |
             ________________|_|__________________________________
            |  _________________________________________________  |
            | |                 |_____________|                 | |
            | |       ___       ,-------------.                 | |
            | |      | w |      |_____________|    Nintendo     | |
            | |   ___|   |___   ,-------------.                 | |
            | |  | a  ,-.  d |  |SELECT  START|   ___B  ___A    | |
            | |  |___ `-' ___|  ;=============`  |,-.| |,-.|    | |
            | |      | s |      |  ===   ===  |  |.o,| |.i,|    | |
            | |      |___|      ;==tab==Return;  '---' '---'    | |
            | |_________________|_____________|_________________| |
            |_____________________________________________________|

    w for up, a for left, s or down, d for right, tab for "select", return for "start", o for "B" and i for "A"
    to modify this mapping, change the optional parameters when initializing the keybind class the data should be given as abs
    two items list for example, to map the left button: Left = [VK_code, Scan_code]´
