from win32api import GetKeyState, keybd_event
import multiprocessing as mltpr
import win32com.client as comclt
import matplotlib.pyplot as plt
import win32con, math, time
import matplotlib, numpy
import os, pyautogui, pickle
from Models import *
from Config import *
from Utils import *
import keras
from keras import backend
from keras.callbacks import *
from keras.models import Sequential, load_model
from keras.layers import *
from keras import backend as K
from mss import mss
from IPython.display import clear_output
import os
import tensorflow as tf


####################################
##             Classes            ##
####################################

class PlotLosses(keras.callbacks.Callback):
    def __init__(self, logs={}):

        ##############################
        # Attributes initializations #
        ##############################

        plt.ion() # Interactive mode on
        self.fig = plt.figure()
        self.i = 0 # Just a counter to be able to plot
        self.x = [] # A list of the successive self.i
        self.losses = [] # Log of the loss
        self.DirectionAccuracy = [] #log of the direction prediction accuracy
        self.ABAccuracy = [] #log of the A-B buttons prediction accuracy
        self.ValDirectionAccuracy = [] #log of the direction prediction accuracy
        self.ValABAccuracy = [] #log of the A-B buttons prediction accuracy
        self.logs = [] #Variable to collect the logs from the model output

        ########################
        # Plot initializations #
        ########################

        self.axLoss = self.fig.add_subplot(211)
        self.axAcc = self.fig.add_subplot(212)
        self.LossLine, = self.axLoss.plot(self.losses,'r', label="Loss function")
        self.axLoss.legend()

        self.AccDirLine, = self.axAcc.plot(self.DirectionAccuracy,'b', label="Direction accuracy")
        self.AccABLine, = self.axAcc.plot(self.ABAccuracy,'g', label="A and B accuracy")

        self.ValAccDirLine, = self.axAcc.plot(self.ValDirectionAccuracy,'b--', label="Val. Direction accuracy")
        self.ValAccABLine, = self.axAcc.plot(self.ABAccuracy,'g--', label="Val. A and B accuracy")

        self.axAcc.legend()

        # Draw the first version of the window
        self.fig.canvas.draw()
        self.fig.canvas.start_event_loop(0.001)

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs) # Collect the logs from the batch
        self.x.append(self.i) # Simply append the last i to the x list
        self.losses.append(logs.get('loss'))
        self.DirectionAccuracy.append(logs.get('dense_5_binary_accuracy'))
        self.ValDirectionAccuracy.append(logs.get('val_dense_5_binary_accuracy'))
        self.ValABAccuracy.append(logs.get('val_dense_13_binary_accuracy'))
        self.ABAccuracy.append(logs.get('dense_13_binary_accuracy'))
        self.i += 1

        #clear_output(wait=True)
        self.LossLine.set_data(self.x,self.losses)

        self.AccDirLine.set_data(self.x,self.DirectionAccuracy)
        self.AccABLine.set_data(self.x,self.ABAccuracy)

        self.ValAccDirLine.set_data(self.x,self.ValDirectionAccuracy)
        self.ValAccABLine.set_data(self.x,self.ValABAccuracy)

        self.axLoss.relim()        # Recalculate limits
        self.axLoss.autoscale_view(True,True,True) #Autoscale
        self.axAcc.relim()        # Recalculate limits
        self.axAcc.autoscale_view(True,True,True) #Autoscale
        self.fig.canvas.blit()
        self.fig.canvas.start_event_loop(0.001)

#    def on_train_end(self,logs):
#        plt.savefig("LastLossPlot")
#        plt.close()



class DataSet():

    """

    This class is used to store a dataset in order to save it through pickle.
    it has two attributes:

    name ---------> A string giving the name of the dataset (used to create the file)
    FrameSet -----> The set of recorded frames
    InputSet -----> The set of recorded inputs<
    Proprocessed -> Bool to determine if the data have been Preprocessed or not

    """

    def __init__(self,name,FrameSet=[],InputSet=[]):
        self.name = name # Name of the data set, this will be used to save the dataset as "name.p"
        self.FrameSet = FrameSet # Set of frames (numpy arrays)
        self.InputSet = InputSet # Set of inputs (8x1 vector)
        self.Proprocessed = False

    def save(self):
        """

        Save the dataset in a pickle file. The target directory is set to ./DataSet/
        if this directory is not available, the program will create one.

        """
        os.system("mkdir DataSets")
        numpy.save("DataSets\\"+self.name+"Frames",self.FrameSet)
        numpy.save("DataSets\\"+self.name+"Inputs",self.InputSet)


    def load(self):
        """
        Loads the dataset by looking in the dataset folder
        This method use self.name to determine which dataset to load

        """
        try: # check if the file really exists
            self.FrameSet = numpy.load("DataSets\\"+self.name+"Frames.npy")
        except:
            print("Error laoding the frames, please check the name of your dataset or \nsee if your files are not corrupted")

        try: # check if the file really exists
            self.InputSet = numpy.load("DataSets\\"+self.name+"Inputs.npy")
        except:
            print("Error loading the inputs, please check the name of your dataset or \nsee if your files are not corrupted")

    def Unload(self):
        """
        I had to introduce this method to avoid oom errors while giving multipledataset as entry
        this is used when the user gives a series of datasets in a file. it basically remove the
        data in the dataset by assigning empty lists to the dataset attributes

        """
        self.FrameSet = []
        self.InputSet = []
        self.PreFrames = []
        self.EvenedFrameSet = []
        self.EvenedInputSet = []

    def printrandom(self,number_of_prints):
        """
        This methods shows "number_of_prints" frames
        and print the associated inputs in the console.
        This can be used to check the consistency of the dataset
        """
        # check if the data have been Preprocessed, or not yet
        if self.Proprocessed == False:
            self.PreProcess()

        # ramdomly pick "number_of_prints" frames
        # and shows them to the user using matplotlib's imshow()
        for i in range(number_of_prints):
            Index = np.random.randint(len(self.FrameSet))
            print(str(self.InputSet[Index])) # prints the input in the console
            plt.imshow(self.PreFrames[Index])
            plt.show()

    def min_not_Zero(self,npList):
        Max = numpy.max(npList)
        Min = Max
        for i in npList:
            if i != 0:
                if i < Min:
                    Min = i
        return Min

    def PreProcess(self,Verbose = False):
        temp0 = np.empty(self.FrameSet[0].shape)
        temp1 = np.empty(self.FrameSet[0].shape)
        temp2 = np.empty(self.FrameSet[0].shape)
        temp3 = np.empty(self.FrameSet[0].shape)
        PreprocessedFrames = []
        for i in range(len(self.FrameSet)):
            temp3 = temp2
            temp2 = temp1
            temp1 = temp0
            temp0 = self.FrameSet[i]
            PreprocessedFrames.append(numpy.concatenate((temp0,temp1,temp2,temp3)))
        self.PreFrames = numpy.array(PreprocessedFrames)
        if Verbose:
            print(self.PreFrames.shape)
        self.Proprocessed = True





    def EvenKey(self):
        self.PreProcess()
        InputSum = numpy.sum(self.InputSet,axis=0)
        print("Before: "+str(InputSum))
        MaxTimeKeyPressed = self.min_not_Zero(InputSum)
        self.EvenedFrameSet = []
        self.EvenedInputSet = []
        for key in range(len(self.InputSet[0])):
            count = 0
            if InputSum[key] > 0:
                Frame = 0
                while count < MaxTimeKeyPressed:
                    if self.InputSet[Frame,key] == 1:
                        self.EvenedFrameSet.append(self.PreFrames[Frame])
                        self.EvenedInputSet.append(self.InputSet[Frame])
                        count += 1
                    Frame += 1
        InputSum = numpy.sum(self.EvenedInputSet,axis=0)
        print("After: "+str(InputSum))

    def Merge(self,Name):
        Lenght1 = len(self.FrameSet)
        TempFrameSet = numpy.load("DataSets\\"+Name+"Frames.npy")
        TempInputSet = numpy.load("DataSets\\"+Name+"Inputs.npy")
        Lenght2 = len(TempFrameSet)
        self.FrameSet = numpy.concatenate((self.FrameSet,TempFrameSet))
        self.InputSet = numpy.concatenate((self.InputSet,TempInputSet))
        if (Lenght1+Lenght2) != len(self.FrameSet):
            print("Error in Merging")
        print(len(self.FrameSet))

class Key:
    """ This class corresponds to the keyboard key, it can be used to
        monitor a specific key or send this key to the active application

        The key class currently has 5 methods: __init(self,name,code,HardwCode1,HardwCode2);
        getstatus(self); GenStr(self); Press(self); Release(self);

        its attributes are:
        - Name -------> string to name the key, this is not used and kept at the user's preference
        - Code -------> This is the virtual key code stored as an Hexadecimal values, the code list can be found
                        here: https://docs.microsoft.com/en-us/windows/desktop/inputdev/virtual-key-codes
        - HardwCode --> This is the scan code of the key the code list can be found here:
                        http://www.gamespp.com/directx/directInputKeyboardScanCodes.html
        - status -----> a binary attribute, 0 if the key is not pressed, 1 if it is

    """

    ##########
    ## Init ##
    ##########
    def __init__(self,name,code,HardwCode):
        self.name = name #Name of the key, not used anymore, kept at user's preference
        self.code = code #Virtual key code
        self.HardwCodePress = HardwCode #Scan code
        self.status = 0 # zero if the key is not pressed

    #############
    ## Methods ##
    #############

    def getstatus(self):
        """

        This method is used to get the key status (pressed or not),
        I had to add the arbitrary >100 test because GetKeyState
        returns 1 or 0 hen the key is not pressed (alternating between each press)
        and 128 or 127 if it is pressed (alternating between each press)

        """
        RawVal = GetKeyState(self.code)
        if abs(RawVal)>100:
            self.status = 1 #The key is pressed
        else:
            self.status = 0 #The key is not pressed
        return RawVal

    def GenStr(self):
        """

        This method was used to debug the program, it returns a string
        giving the name of the key and if it is pressed or not

        """
        str = ""
        str += self.name
        str += " "
        if self.status == 0:
            str += "released    "
        if self.status == 1:
            str += "pressed    "
        return str

    def Press(self):
        keybd_event(self.code,self.HardwCodePress,0,0)
        #pyautogui.keyDown(self.name)

    def Release(self):
        keybd_event(self.code,self.HardwCodePress, win32con.KEYEVENTF_KEYUP,0)
        #pyautogui.keyUp(self.name)


class KeyBinds:

    """
    Serves as interface between the program and the emulator

    The default keybinds are mapped for an azerty keyboard as show on the ascii bellow.

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
    to modify this mapping, change the optional parameters when initializing the class the data should be given as abs
    two items list for example, to map the left button: Left = [VK_code, Scan_code]

    you can find lists of VK_codes and scan codes here:
    - VK_Code -------> This is the virtual key code stored as an Hexadecimal values, the code list can be found
                        here: https://docs.microsoft.com/en-us/windows/desktop/inputdev/virtual-key-codes
    - Scan Code -----> This is the scan code of the key the code list can be found here:
                        http://www.gamespp.com/directx/directInputKeyboardScanCodes.html

    This class has 9 attributes:
    - LeftKey -------> The key binded to the left direction on the controller (default "a");
    - RightKey ------> The key binded to the Right direction on the controller (default "d");
    - DownKey -------> The key binded to the Down direction on the controller (default "s");
    - UpKey ---------> The key binded to the Up direction on the controller (default "w");
    - A -------------> The key binded to the A button (default "i");
    - B -------------> The key binded to the B button (default "o");
    - Start ---------> The key binded to the Start button (default "tab");
    - Select --------> The key binded to the Select button (default "return");
    - KeyTable ------> List of all the keys

    """

    ##########
    ## Init ##
    ##########
    def __init__(self,Left=[0x41,0x1E],Right=[0x44,0x20],Down=[0x53,0x1F],Up=[0x57,0x11],A=[0x49,0x17],B=[0x4F,0x18],Start=[0x0D,0x1C],Select=[0x09,0x0F]):
        self.LeftKey = Key("q",Left[0],Left[1])
        self.RightKey = Key("d",Right[0],Right[1])
        self.DownKey = Key("s",Down[0],Down[1])
        self.Up = Key("z",Up[0],Up[1])
        self.A = Key("i",A[0],A[1])
        self.B = Key("o",B[0],B[1])
        #self.Start = Key("return",Start[0],Start[1])
        #self.Select = Key("tab",Select[0],Select[1])
        self.KeyTable = [self.LeftKey,self.RightKey,self.DownKey,self.Up,self.A,self.B]


    #############
    ## Methods ##
    #############
    def EvaluateStatus(self):
        """

        updates the status of all the keys

        """
        temp = [i.getstatus() for i in self.KeyTable]
        return 0

    def PrintStatus(self,Verbose=True,Time = 0.0):
        """

        Used to monitor the current states of all the keys, if Verbose is
        set to True, it will display the input vector in the temrinal.
        time is an optional value that used to be used to validate the framerate

        """

        StatusVect = [i.status for i in self.KeyTable] #simply writes the status in a comprehensive list
        StatusVect.append(Time)
        if Verbose == True:
            print(StatusVect, end='\r', flush=True)
        return StatusVect

    def SendInput(self,inputVector,FPS):
        """

        Sends the inputs to the activ window
        The input should be given as a vector matching the shape of Keybinds.keytable

        """
        t0 = time.time() #initial time
        t1 = time.time() #current time
        while (t1-t0)<(1.0/float(FPS)): #loop that sends the input each 10th of a frame
            for i in range(len(inputVector)):
                if inputVector[i]==1:
                    self.KeyTable[i].Press()
                if inputVector[i]==0:
                    self.KeyTable[i].Release()
            time.sleep(1.0/(10*FPS))
            t1 = time.time()
        return 0


    def zeroInput(self):
        """

        Resets all the status to zero (avoid pressing an key when using this function)

        """
        for i in self.KeyTable:
            i.getstatus()


    def Botplay(self,Model,scale,FPS=FPS,RecordKeyToggle=0x20,RecordKeyHW=0x39):
        ToggleKey = Key("Toggle",RecordKeyToggle,RecordKeyHW)
        sample = TakeImage(scale,Print=False)
        self.zeroInput()
        print("Waiting for toggle key press")
        while ToggleKey.status == 0:
            ToggleKey.getstatus()
        time.sleep(0.1)
        ToggleKey.getstatus()
        factor = 1.0
        print("Begin of the playing session\n")
        temp0 = np.empty(sample.shape)
        temp1 = np.empty(sample.shape)
        temp2 = np.empty(sample.shape)
        temp3 = np.empty(sample.shape)
        while ToggleKey.status == 0:
            ToggleKey.getstatus()
            temp3 = temp2
            temp2 = temp1
            temp1 = temp0
            temp0 = TakeImage(scale)
            Screen = numpy.concatenate((temp0,temp1,temp2,temp3))
            Vector = Model.predict(numpy.array([Screen,]))
            Vector = numpy.concatenate((Vector[0],Vector[1]),axis = 1)
            Test.SendInput(numpy.round((Vector[0])),FPS)
            #print("  input: "+str(numpy.round(Vector[0]+0.1)))
            time.sleep(float(factor)/float(FPS))
            ToggleKey.getstatus()
        print("End of the playing session")
        return 0
