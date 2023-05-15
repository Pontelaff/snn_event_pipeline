import numpy as np
from layers import ConvLayer
from timeit import timeit
from dataloader import loadKernels, loadEvents
from utils import SpikeQueue


MODEL_PATH = "pretrained/LIFFireNet.pth"
INPUT_PATH = "datasets/cup-drop-short.h5"
NUM_INPUT = 2000000

SEG_WIDTH = 32
SEG_HEIGHT = 32
REC_LAYERS = (0,3)

# initialise kernel weights
inputKernels, hiddenKernels, recKernels, outputKernels = loadKernels(MODEL_PATH)

# define the structured data type
dtype = np.dtype([('u', np.float16), ('t', np.int32)])
# initialise neuron states
numHiddenLayers = len(hiddenKernels)
inputNeurons = np.zeros([len(inputKernels), SEG_WIDTH, SEG_HEIGHT], dtype=dtype)
hiddenNeurons = np.zeros([numHiddenLayers, len(hiddenKernels[0]), SEG_WIDTH, SEG_HEIGHT], dtype=dtype)
outputNeurons = np.zeros([len(outputKernels), SEG_WIDTH, SEG_HEIGHT], dtype=dtype)

# load input events from file
eventInput = loadEvents(INPUT_PATH, NUM_INPUT)

def inference(inputNeurons, hiddenNeurons, inputKernels, hiddenKernels, eventInput):
    # init layers
    inputLayer = ConvLayer(len(inputKernels[0]), len(hiddenKernels[0]), len(hiddenKernels[0, 0, 0]), dtype)
    convLayer = ConvLayer(len(hiddenKernels[0, 0]), len(hiddenKernels[0]), len(hiddenKernels[0, 0, 0]), dtype)

    # run input layer
    inputLayer.assignLayer(eventInput, inputKernels, inputNeurons)
    inputNeurons, ffQ, _ = inputLayer.forward()
    print("%d spikes in input layer" %(len(ffQ)))
    num_spikes = len(ffQ)

    recQueues = [SpikeQueue() for _ in range(len(REC_LAYERS))]

    # run hidden layers
    for l in range(numHiddenLayers):
        try:
            recInd = REC_LAYERS.index(l)
            rec = True
        except ValueError as ve:
            rec = False

        if rec:
            convLayer.assignLayer(ffQ, hiddenKernels[l], hiddenNeurons[l], recQueues[recInd], recKernels[recInd])
            hiddenNeurons[l], ffQ, recQueues[recInd]= convLayer.forward()
        else:
            convLayer.assignLayer(ffQ, hiddenKernels[l], hiddenNeurons[l])
            hiddenNeurons[l], ffQ, _ = convLayer.forward()

        print("%d spikes in layer %d" %(len(ffQ), l+1))
        num_spikes += len(ffQ)

    return num_spikes

runs=1
time = timeit(lambda: inference(inputNeurons, hiddenNeurons, inputKernels, hiddenKernels, eventInput), number=runs)
print(f"Time: {time/runs:.6f}")

#print(" c  x  y  t")
# while outQ.qsize() > 0:
#     event = outQ.get()
#     #print("%2d %2d %2d %2d" % (event.channel, event.x_pos, event.y_pos, event.timestamp))

pass