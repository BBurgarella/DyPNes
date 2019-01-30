from win32api import GetKeyState, keybd_event
import multiprocessing as mltpr
import win32com.client as comclt
import matplotlib.pyplot as plt
import win32con, math, time
import matplotlib, numpy
import os, pyautogui, pickle
from Models import *
from Config import *
import Classes
from Classes import *

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


def DoubleListShuffle(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]


def TakeImage(scale,Print=False):
    with mss() as sct:
        img = numpy.array(sct.grab({"top": top, "left": left, "width": width, "height": height}))
        img = img[::scale,::scale,0:3]
        #imgR = img[::scale,::scale,0]
        #imgG = img[::scale,::scale,1]
        #imgB = img[::scale,::scale,2]
        #img = (imgR+imgG+imgB)/765
        img = img/255
        if Print:
            plt.imshow(img)
            plt.show()
        return img

def DataGenerator(FileName,Evened=False):
    File = open(FileName,'r')
    lines = File.read().splitlines()
    count = 0
    for DataName in lines:
        ForTrainingDataSet = Classes.DataSet(DataName)
        ForTrainingDataSet.load()
        ForTrainingDataSet.PreProcess()
        if Evened == True:
            PreviousDataSet.EvenKey()
        ScreenList = ForTrainingDataSet.PreFrames
        VectorList = ForTrainingDataSet.InputSet
        X, Y = numpy.array(ScreenList), numpy.array(VectorList)
        yield X,{'dense_5': Y[:,:4], 'dense_13': Y[:,4:6]},DataName,len(lines)

#####################################
##             Functions           ##
#####################################

def PlotValid(model,DataSetObj):
    DataSetObj.load()
    DataSetObj.EvenKey()
    DataSetObj.PreProcess()
    DataSetObj.EvenKey()
    ScreenList = DataSetObj.EvenedFrameSet
    VectorList = DataSetObj.EvenedInputSet
    #X, Y = DoubleListShuffle(numpy.array(ScreenList),numpy.array(VectorList))
    X, Y = numpy.array(ScreenList), numpy.array(VectorList)
    Y2 = model.predict(X)
    Y2 = numpy.concatenate((Y2[0],Y2[1]),axis = 1)
    newfig = plt.figure(figsize = (2,1))
    axY = newfig.add_subplot(111)
    axY2 = newfig.add_subplot(212)
    axY.imshow(Y,aspect="auto")
    axY2.imshow(Y2,aspect="auto")
    plt.savefig(DataSetObj.name)

def Train(model,KeyBinds,scale,plot_losses,FPS=FPS,RecordKeyToggle=0x20,RecordKeyHW=0x39,PreviousDataSet=None,Save=True,epochs=1,batch_size=20,UseEvenedData = False):

    """
    Use this function to train a model,

    # mandatory inputs:
    1 - a model (keras object)
    2 - A keybind object
    3 - a scale (to reduce the size of the images)

    # Optional inputs
    1 - FPS (Default is set by the Config.py file)
    2 - RecordKeyToggle Virtual key code (Default is the spacebar)
    3 - RecordKeyToggle hardware code (Default is the spacebar)
    4 - PreviousDataSet this can be either a list of datasets or a dataset (default is None)
    5 - Save, bool to determine if the dataset should be saved or not (only used if Previous Dataset is None)
    6 - epochs, number of epochs for the keras fit call (default is 1)
    7 - batch_size, batch size for the keras fit call (default is 20)
    8 - UseEvenedData bool to determine if the data should be evened or not (default is False because I had not enough data at the begining)

    This function returns nothing, it only changes the weights and bias in the model to follow the given dataset

    """

    # if there is no data set provided, record a new one
    if PreviousDataSet == None:
        ToggleKey = Classes.Key("Toggle",RecordKeyToggle,RecordKeyHW) #Default toggle key is the spacebar
        print("Waiting for toggle key press")
        t0 = time.time()
        while ToggleKey.status == 0:
            ToggleKey.getstatus()
            t1 = time.time()
            print("I have waited for "+str(t1-t0)+" seconds              ", end = "\r")

        # Need to add some time for the user to prevent the computer from
        # instantly beliving that the spacebar have been pressed again
        time.sleep(0.5)
        ToggleKey.getstatus()
        print("Begining Record")
        t0 = time.time()
        ScreenList = []
        VectorList = []

        # Main loop during record
        while ToggleKey.status == 0:
            Screen = TakeImage(scale)
            ScreenList.append(Screen)
            KeyBinds.EvaluateStatus()
            time.sleep(1.0/float(FPS)) # Wait for one frame
            t1 = time.time()
            Vector = KeyBinds.PrintStatus(Time = t1-t0)
            ToggleKey.getstatus()
            VectorList.append(Vector[0:6])

        # Plot of all the recorded Input
        # This is not that necessary, I might remove it
        plt.imshow(VectorList,aspect="auto")
        plt.show()

        # If the optional parameter save is true (default) then the user will be asked
        # to name the newly created dataset which will then be saved
        if Save:
            dataname=input("What is the name of the dataset you want to record ?\n")
            Data = Classes.DataSet(dataname,numpy.array(ScreenList),numpy.array(VectorList))
            Data.save()

        # if save is false, a dataset without name is created
        else:
            Data = Classes.DataSet("NONAME",numpy.array(ScreenList),numpy.array(VectorList))

        # Preprocessing of the dataset
        Data.PreProcess()
        ScreenList = Data.PreFrames

        if UseEvenedData == True:
            PreviousDataSet.EvenKey() #This function tries to show as many different situations as possible to the model
            VectorList = Data.EvenedInputSet

        else:
            VectorList = Data.InputSet

    # The user can enter the dataset as a list of datasets, if so, each dataset is parsed sequentially
    # I need to add a function to pick data from every dataset to even everything
    elif isinstance(PreviousDataSet, str):
        for i in range(epchs):
            Rate = K.get_value(model.optimizer.lr)
            print("Epoch {}/{}, Current learning rate: {}            \n".format(i,epchs,Rate))
            if UseEvenedData == True: #same as before, tries to show as many different situations as possible to the model
                count = 1
                for Data in DataGenerator(PreviousDataSet,Evened=True):
                    print("Currently reading data from '{}', which is the dataset #{}/{}      ".format(Data[2],count,Data[3]), end = "\r")
                    count += 1
                    X, Y = Data[0], Data[1]+0.000001
                    model.fit(X,Y,validation_split=0,epochs=1,verbose = 0,callbacks=[plot_losses,TerminateOnNaN(),ModelCheckpoint("Models\\weights.{epoch:02d}.hdf5")],batch_size=batch_size)

            else:
                count = 1
                for Data in DataGenerator(PreviousDataSet,Evened=False):
                    print("Currently reading data from '{}', which is the dataset #{}/{}      ".format(Data[2],count,Data[3]), end = "\r")
                    count += 1
                    X, Y = Data[0], Data[1]
                    model.fit(X,Y,validation_split=0,epochs=1,verbose = 0,callbacks=[plot_losses,TerminateOnNaN(),ModelCheckpoint("Model\\weights.{epoch:02d}.hdf5")],batch_size=batch_size)
            model.save("Models\\SaveModel_Epoch{}.h5".format(i))
            K.set_value(model.optimizer.lr,float(Rate-decayRate))
        return 0 # Just to end the function


    else:
        if UseEvenedData == True:
            PreviousDataSet.PreProcess() # concatenate 4 frames together
            PreviousDataSet.EvenKey() # same as before, tries to show as many different situations as possible to the model
            ScreenList = PreviousDataSet.EvenedFrameSet
            VectorList = PreviousDataSet.EvenedInputSet
        else:
            PreviousDataSet.PreProcess() # concatenate 4 frames together
            ScreenList = PreviousDataSet.PreFrames
            VectorList = PreviousDataSet.InputSet

    X, Y = numpy.array(ScreenList), numpy.array(VectorList)

    # [Y[:,:4] corresponds to the directions
    # Y[:,4:6] to the A - B buttons
    model.fit(X,[Y[:,:4],Y[:,4:6]],validation_split=0,epochs=epochs,callbacks=[plot_losses,keras.callbacks.TerminateOnNaN(),ModelCheckpoint("Models\\weights.{epoch:02d}.hdf5")],batch_size=batch_size)
    print("End of the training session")
    return 0 # Just to end the function
