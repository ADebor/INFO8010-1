# **********************************
# *      INFO8010 - Project        *
# *  CycleGAN for style transfer   *
# *             ---                *
# *  Antoine DEBOR & Pierre NAVEZ  *
# *       ULiège, May 2021         *
# **********************************

import torch
import torchvision.transforms as transforms
from custom_transforms import AddGaussianNoise

# MODE
######
MODE = 'H2Z'
#MODE = 'F2V'
#MODE = 'F2I'


# LEARNING PARAMETERS
#####################
N_EPOCHS = 100
LR = 2e-4
ADV_COEFF =  1
CYCLE_COEFF = 10
IDENTITY_COEFF = 0.5
GEN_LOSS_COEFF = ADV_COEFF, CYCLE_COEFF, IDENTITY_COEFF
BETAS = (0.5, 0.999)

# DEVICE
########
DEVICE = 'cuda:0' if torch.cuda.is_available()  else 'cpu'

# IMAGE PARAMETERS
##################
IMG_DIM = 128

# Non-unitary normalization factor (deprecated)
#L1_NORM_FACTOR = 3*IMG_DIM**2
# Unitary factor
L1_NORM_FACTOR = 1
TRANSFORM_LIST = [transforms.ToTensor(),
                  transforms.Resize((IMG_DIM, IMG_DIM)),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                  transforms.RandomHorizontalFlip(p=1),
                  transforms.RandomRotation(degrees=20, interpolation=transforms.InterpolationMode.BILINEAR),
                  transforms.RandomAffine(degrees=20)]

# SAVING PARAMETERS
###################
SAVE_PATH = "/content/gdrive/My Drive/Deep Learning/Project/saved_images_cycleGAN_"+MODE+"/" # change with your own path !
SAVE_MODEL_PATH = "/content/gdrive/My Drive/Deep Learning/Project/saved_models_cycleGAN_"+MODE+".pth"

# DATASET PARAMETERS
####################

ZIP_PATH = '/content/gdrive/My Drive/Deep Learning/Project/Data/Zip/'   # change with your own path !

# X
if MODE == 'H2Z':
    TRAIN_X_ID = 'trainHorse'
    TEST_X_ID = 'testHorse'
else:
    TRAIN_X_ID = 'trainFaces'
    TEST_X_ID = 'testFaces'

# Y
if MODE == 'H2Z':
    TRAIN_Y_ID = 'trainZebra'
    TEST_Y_ID = 'testZebra'

elif MODE == 'F2V':
    TRAIN_Y_ID = 'trainPaintings'
    TEST_Y_ID = 'testPaintings'
    ARTIST = 'Vincent_VanGogh'

else:
    TRAIN_Y_ID = 'trainPostimpress'
    TEST_Y_ID = 'testPostimpress'
