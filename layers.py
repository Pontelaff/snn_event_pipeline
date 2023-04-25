import numpy as np
from utils import Spike, Event, EventQueue
from neurocore import Neurocore


KERNEL_SIZE = 3
CONV_CHANNELS = 32
INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 2
SEG_WIDTH = 32
SEG_HEIGHT = 32

class Layer:
    def __init__(self, inChannels, inQueue, kernels, neurons, layer = 1) -> None:
        numKernels = len(kernels)
        kernelSize = len(kernels[0,0])
        self.layer = layer
        self.neurons = neurons
        # generate neurocores
        self.neurocores = [Neurocore(c, numKernels, kernelSize) for c in range(inChannels)]
        self.inQueue = inQueue
        self.outQueue = EventQueue(self.layer)



# initialise neuron states
inputNeurons = np.zeros([CONV_CHANNELS, SEG_WIDTH, SEG_HEIGHT, 2], dtype=np.float16)
hiddenNeurons = np.zeros([6, CONV_CHANNELS, SEG_WIDTH, SEG_HEIGHT, 2], dtype=np.float16)
outputNeurons = np.zeros([SEG_WIDTH, SEG_HEIGHT, 2], dtype=np.float16)

# initialise kernel weights
inputKernels = np.random.rand(CONV_CHANNELS, INPUT_CHANNELS, KERNEL_SIZE, KERNEL_SIZE).astype(np.float16)
hiddenKernels = np.random.rand(6, CONV_CHANNELS, CONV_CHANNELS, KERNEL_SIZE, KERNEL_SIZE).astype(np.float16)
outputKernels = np.random.rand(OUTPUT_CHANNELS, CONV_CHANNELS, CONV_CHANNELS, 1, 1).astype(np.float16)

# initialise event queues
eventInput = EventQueue(-1)
inQueue = EventQueue(0)
outQueue = EventQueue(1)



inputLayer = Layer(INPUT_CHANNELS, eventInput, inputKernels, inputNeurons, 0)
convLayer = Layer(CONV_CHANNELS, inQueue, hiddenKernels[0], hiddenNeurons[0], 1)
spike = Spike(0,0,12)

# pad each channel with zeros (don't pad neuron states)
layer1Neurons = np.pad(hiddenNeurons[0], ((0,0),(1,1),(1,1),(0,0)), 'constant')

# recognizable test values
hiddenKernels[0,0,0,0] = [1,2,3]
hiddenKernels[0,1,0,0] = [44,55,66]
hiddenKernels[0,2,0,1] = [0,55,0]
layer1Neurons[0,11,23:26] = [[0,1],[2,3],[55,10]]

nc = Neurocore(0, CONV_CHANNELS, KERNEL_SIZE)
nc.assignLayer(1, hiddenKernels)
nc.loadNeurons(spike, layer1Neurons)

nc.leakNeurons()
nc.applyKernel()
outQ = nc.checkTreshold()

pass