"""
Use this file to configure the model to your personal setup

"""

## The following variables are used to determine where to take the screenshots
## if you use the default scaling in MESEN and put the window on the
## upper- left corner of your screen this should be working (my resolution is 1920x1080)
top=55 # vertical position of the top-left point
left=32 # Horizontal position of the top-left point
width=250 # width of the screenshots
height=224 # height of the screenshots

###############################################################################

FPS = 60 # number of frames per seconds (be carefull, your computer might not be able to be fast enough for high values)
Emulator = "Mesen" # name of the emulator window

###############################################################################

# Factor by which the screenshots are downscaled
Scale = 2

# learning rate of the optimizer
LearningRate = 0.001

 # number of epochs per dataset
epchs =  10

# learning rate decay
decayRate = LearningRate / epchs

 # batch size used when calling the keras fit function
btch_sz = 10

# Time spend training, once the traning time is larger than this, the program will
# stop looping on the datasets (the current "fit" function will finish though)
MaxTime = 1
