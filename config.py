############################
## Configuration Parameters
############################

# FireNet Parameters
SEG_WIDTH = 48
SEG_HEIGHT = 24
# Index of recurrent layers
REC_LAYERS = (1,4)

# Time window for one forward pass
EVENT_TIMESLICE = 500

# Dataloader
MODEL_PATH = "pretrained/LIFFireNet.pth"
INPUT_PATH = "datasets/cup-drop-400.h5"
NUM_INPUT = -1
