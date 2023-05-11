import numpy as np
from layers import ConvLayer
from timeit import timeit
from dataloader import loadKernels, loadEvents


MODEL_PATH = "pretrained/LIFFireNet.pth"
INPUT_PATH = "datasets/cup-drop-short.h5"
NUM_INPUT = 2000000

CONV_CHANNELS = 32
KERNEL_NUM = CONV_CHANNELS
HIDDEN_LAYERS = 6
INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 2
SEG_WIDTH = 32
SEG_HEIGHT = 32

# initialise neuron states
inputNeurons = np.zeros([CONV_CHANNELS, SEG_WIDTH, SEG_HEIGHT, 2], dtype=np.float16)
hiddenNeurons = np.zeros([6, CONV_CHANNELS, SEG_WIDTH, SEG_HEIGHT, 2], dtype=np.float16)
outputNeurons = np.zeros([SEG_WIDTH, SEG_HEIGHT, 2], dtype=np.float16)

# initialise kernel weights
inputKernels, hiddenKernels, recKernels, outputKernels = loadKernels(MODEL_PATH)

# load input events from file
eventInput = loadEvents(INPUT_PATH, NUM_INPUT)

layerTimestamps = np.zeros(8)
def inference(inputNeurons, hiddenNeurons, inputKernels, hiddenKernels, eventInput):
    # init layers
    inputLayer = ConvLayer(2, len(hiddenKernels[0]), len(hiddenKernels[0, 0, 0]))
    convLayer = ConvLayer(CONV_CHANNELS, len(hiddenKernels[0]), len(hiddenKernels[0, 0, 0]))

    inputLayer.assignLayer(eventInput, inputKernels, inputNeurons, layerTimestamps[0], False)
    inputNeurons , ffQ, layerTimestamps[0] = inputLayer.forward()
    print("%d spikes in input layer" %(len(ffQ)))
    num_spikes = len(ffQ)

    for l in range(HIDDEN_LAYERS):
        convLayer.assignLayer(ffQ, hiddenKernels[l], hiddenNeurons[l], layerTimestamps[l+1], False)
        hiddenNeurons[l], ffQ, layerTimestamps[l+1] = convLayer.forward()
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