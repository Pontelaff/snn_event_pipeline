import numpy as np
from utils import Event, EventQueue
from layers import ConvLayer
from timeit import timeit
import copy as cp


KERNEL_SIZE = 3
CONV_CHANNELS = 32
KERNEL_NUM = CONV_CHANNELS
HIDDEN_LAYERS = 6
INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 2
SEG_WIDTH = 32
SEG_HEIGHT = 32
INPUT_EVENTS = 1000


# recognizable test values
#hiddenKernels[0,0,0,0] = [1,2,3]
#hiddenKernels[0,1,0,0] = [44,55,66]
#hiddenKernels[0,2,0,1] = [-55,55,-55]
#hiddenNeurons[0,0,11,23:26] = [[0,1],[2,3],[55,10]]

def inference():

    rng = np.random.default_rng(12)

    # initialise neuron states
    inputNeurons = np.zeros([CONV_CHANNELS, SEG_WIDTH, SEG_HEIGHT, 2], dtype=np.float16)
    hiddenNeurons = np.zeros([6, CONV_CHANNELS, SEG_WIDTH, SEG_HEIGHT, 2], dtype=np.float16)
    outputNeurons = np.zeros([SEG_WIDTH, SEG_HEIGHT, 2], dtype=np.float16)

    # initialise kernel weights
    inputKernels = rng.random((KERNEL_NUM, INPUT_CHANNELS, KERNEL_SIZE, KERNEL_SIZE)).astype(np.float16)-0.48
    hiddenKernels = rng.random((6, KERNEL_NUM, CONV_CHANNELS, KERNEL_SIZE, KERNEL_SIZE)).astype(np.float16)-0.5
    outputKernels = rng.random((OUTPUT_CHANNELS, CONV_CHANNELS, CONV_CHANNELS, 1, 1)).astype(np.float16)*0.5

    # init layers
    inputLayer = ConvLayer(2, len(hiddenKernels[0]), len(hiddenKernels[0, 0, 0]))
    convLayer = ConvLayer(CONV_CHANNELS, len(hiddenKernels[0]), len(hiddenKernels[0, 0, 0]))

    # initialise event queues
    eventInput = EventQueue()

    # slider
    for i in range(SEG_HEIGHT//4*(SEG_WIDTH-1)):
        eventInput.put(Event(i//SEG_HEIGHT, i%SEG_HEIGHT, i//SEG_HEIGHT, 0))
        eventInput.put(Event(i//SEG_HEIGHT+1, i%SEG_HEIGHT, i//SEG_HEIGHT, 1))

    inputLayer.assignLayer(eventInput, inputKernels, inputNeurons, False)
    inputNeurons , ffQ = inputLayer.forward(None)
    print("%d spikes in input layer" %(ffQ.qsize()))

    for l in range(HIDDEN_LAYERS):
        convLayer.assignLayer(ffQ, hiddenKernels[l], hiddenNeurons[l], False)
        hiddenNeurons[l], ffQ = convLayer.forward(None)
        print("%d spikes in layer %d" %(ffQ.qsize(), l+1))

runs = 5
time = timeit(lambda: inference(), number=runs)
print(f"Time: {time/runs:.6f}")

#print(" c  x  y  t")
# while outQ.qsize() > 0:
#     event = outQ.get()
#     #print("%2d %2d %2d %2d" % (event.channel, event.x_pos, event.y_pos, event.timestamp))

pass