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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'






if __name__ == '__main__':
    run = True
    Test = KeyBinds()
    NeuralNet = initBranchedModel(Scale,Verbose = True)
    #NeuralNet = initVerySmallModel(Scale, Verbose = True)
    t0= time.time()
    # Definition of the "plot_losses object" to be able to have the evolution of the loss in real time
    plot_losses = PlotLosses()
    while run == True:
        print("Welcome in DyPBot V0.3\n")
        Choice = input("What would you like to do ?\n1 - Train the bot\n2 - Let the bot play\n3 - Set the training time\n4 - Save the model\n5 - Load a model\n6 - Update the model's learning rate\n7 - Quit\n")
        if Choice == "1":
            Choice2 = input("Would you like to generate a new data set (1) or use an existing one (2)  or use a file (3)?\n")
            if Choice2 == "1":
                t0= time.time()
                Test.zeroInput()
                Train(NeuralNet,Test,Scale,plot_losses,epochs=epchs,batch_size=btch_sz)
                t1 = time.time()
                timestr = str(round(t1-t0,1))
                print("Model Trained in "+timestr+" seconds")
            elif Choice2 == "3":
                FileName = input("please give the name of the file in which the data sets are listed\n")
                t0= time.time()
                t1 = time.time()
                while (t1-t0) < MaxTime:
                    Train(NeuralNet,Test,Scale,plot_losses,Save=False,PreviousDataSet=FileName,epochs=epchs,batch_size=btch_sz)
                    t1 = time.time()
                    timestr = str(round(t1-t0,1))
                    print("Model Trained in "+timestr+" seconds")


            elif Choice2 == "2":
                AddAnother = True
                FirstTime = True
                while AddAnother == True:
                    if FirstTime == True:
                        DataName = input("please give the name of the dataset (without the extension)\n")
                        ForTrainingDataSet = DataSet(DataName)
                        ForTrainingDataSet.load()
                        FirstTime = False
                    else:
                        DataName = input("please give the name of the dataset (without the extension)\n")
                        ForTrainingDataSet.Merge(DataName)
                    Choice3 = input("Would you like to add another dataset ? (1 - yes, 2 - no)\n")
                    if Choice3 == "2":
                        AddAnother = False
                t0= time.time()
                t1 = time.time()
                while (t1-t0) < MaxTime:
                    Train(NeuralNet,Test,Scale,plot_losses,Save=False,PreviousDataSet=ForTrainingDataSet,epochs=epchs,batch_size=btch_sz)
                    t1 = time.time()
                    timestr = str(round(t1-t0,1))
                    print("Model Trained in "+timestr+" seconds")
        if Choice == "2":
            Test.zeroInput()
            Test.Botplay(NeuralNet,Scale)
        if Choice == "3":
            MaxTime = None
            while type(MaxTime) != int:
                MaxTime = input("Enter the maximum training time in s : ")
                try:
                    MaxTime = int(MaxTime)
                except:
                    print("please enter a proper number")
        if Choice == "4":
            ModelName = input("Please give a name to the model\n")
            NeuralNet.save("Models\\"+ModelName+".h5")
        if Choice == "5":
            ModelName = input("Please give the name of the model you want to load\n")
            NeuralNet = load_model("Models\\"+ModelName+".h5")
            print(NeuralNet.summary())
        if Choice == "6":
            print("current learning rate: "+str(backend.get_value(NeuralNet.optimizer.lr)))
            Rate = input("Please enter the new learning rate\n")
            backend.set_value(NeuralNet.optimizer.lr, Rate)
        if Choice == "7":
            run = False
        if Choice == "8":
            # Hidden choice use to test things
            for a in DataGenerator("DataList.txt",Evened=False):
                print(a[0])
