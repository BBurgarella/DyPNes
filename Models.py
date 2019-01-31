from win32api import GetKeyState, keybd_event
import multiprocessing as mltpr
import win32com.client as comclt
import matplotlib.pyplot as plt
import win32con, math, time
import matplotlib, numpy
import os, pyautogui, pickle
from Config import *
from Classes import *
import keras
from Utils import *
from keras.models import Sequential, load_model, Model
from keras.layers import *
from keras.activations import *
from keras.optimizers import *
from keras import metrics
from mss import mss
from IPython.display import clear_output

"""

Use this file to specify your own Models.
I just separated this from the main file to easilly keep things
organised.

"""


def initBranchedModel(scale, Verbose = True):
    sample = TakeImage(scale,Print=False)
    sampleShape = (4*sample.shape[0],sample.shape[1],sample.shape[2])

    #input tensor
    inputs = Input(shape=sampleShape)

    #################################
    #Convolution NN#1 - First layers#
    #################################

    x = Conv2D(128,(3,3))(inputs)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(3,3))(x)

    x = Conv2D(128,(3,3))(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(64,(3,3))(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(64,(3,3))(x)
    x = LeakyReLU()(x)

    x = Conv2D(32,(3,3))(x)
    x = LeakyReLU()(x)

    x = Conv2D(8,(3,3))(x)
    x = LeakyReLU()(x)

    #Convlution Layers output
    ConvOut = Flatten()(x)

    #######################################
    #Dense NN#1 - Predicting the direction#
    #######################################
    x = Dense(64)(ConvOut)
    x = LeakyReLU()(x)

    x = Dense(32)(ConvOut)
    x = LeakyReLU()(x)

    x = Dense(16)(ConvOut)
    x = LeakyReLU()(x)

    x = Dense(12)(ConvOut)
    x = LeakyReLU()(x)

    #Dense #1 Output
    Direction = Dense(4,activation = 'sigmoid')(ConvOut)

    #################################
    #Dense NN#2 - Predicting A and B#
    #################################

    x = concatenate([ConvOut,Direction])

    x = Dense(64)(x)
    x = LeakyReLU()(x)

    x = Dense(32)(x)
    x = LeakyReLU()(x)

    x = Dense(32)(x)
    x = LeakyReLU()(x)

    x = Dense(16)(x)
    x = LeakyReLU()(x)

    x = Dense(16)(x)
    x = LeakyReLU()(x)

    x = Dense(16)(x)
    x = LeakyReLU()(x)

    x = Dense(8)(x)
    x = LeakyReLU()(x)

    #Dense #1 Output
    ABOut = Dense(2,activation = 'sigmoid')(x)

    #Model definition with 1 input and 2 outputs
    model = Model(inputs = [inputs], outputs = [Direction,ABOut])
    model.compile(loss='logcosh', optimizer=SGD(lr=LearningRate,decay = decayRate, nesterov=True),loss_weights=[1, 1],metrics = [metrics.binary_accuracy])

    if Verbose: # if verbose is set to true, this will display a summary of the model
        model.summary()
    return model
