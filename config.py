############################
## Configuration Parameters
############################

import numpy as np

# LIFFireNet simulation parameters
SEG_WIDTH = 48
SEG_HEIGHT = 24
# Architecture of the imported LIFFireNet
FIRENET_LAYERS =("head", "G1", "R1a", "R1b", "G2", "R2a", "R2b", "pred")
# Index of recurrent layers
REC_LAYERS = (1,4)

# Time window for one forward pass
EVENT_TIMESLICE = 500

# Dataloader
MODEL_PATH = "pretrained/LIFFireNet.pth"
INPUT_PATH = "datasets/cup-drop-400.h5"

# Amount of input events (-1 for all)
NUM_INPUT = -1

# Data type for kernels and neuron values
DATATYPE = np.float16
