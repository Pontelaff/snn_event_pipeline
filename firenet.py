import numpy as np
from utils import Event, EventQueue
from layers import ConvLayer
from timeit import timeit


KERNEL_SIZE = 3
CONV_CHANNELS = 32
KERNEL_NUM = CONV_CHANNELS
HIDDEN_LAYERS = 6
INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 2
SEG_WIDTH = 32
SEG_HEIGHT = 32
INPUT_EVENTS = 1000

#Random number generator
rng = np.random.default_rng(12)
# initialise neuron states
inputNeurons = np.zeros([CONV_CHANNELS, SEG_WIDTH, SEG_HEIGHT, 2], dtype=np.float16)
hiddenNeurons = np.zeros([6, CONV_CHANNELS, SEG_WIDTH, SEG_HEIGHT, 2], dtype=np.float16)
outputNeurons = np.zeros([SEG_WIDTH, SEG_HEIGHT, 2], dtype=np.float16)

# initialise kernel weights
inputKernels = rng.random((KERNEL_NUM, INPUT_CHANNELS, KERNEL_SIZE, KERNEL_SIZE)).astype(np.float16)-0.48
hiddenKernels = rng.random((6, KERNEL_NUM, CONV_CHANNELS, KERNEL_SIZE, KERNEL_SIZE)).astype(np.float16)-0.5
outputKernels = rng.random((OUTPUT_CHANNELS, CONV_CHANNELS, CONV_CHANNELS, 1, 1)).astype(np.float16)*0.5

# initialise event queue
eventInput = EventQueue()

# slider
for i in range(SEG_HEIGHT*(SEG_WIDTH-1)//4):
    eventInput.put(Event(i//SEG_HEIGHT, i%SEG_HEIGHT, i//SEG_HEIGHT, 0))
    eventInput.put(Event(i//SEG_HEIGHT+1, i%SEG_HEIGHT, i//SEG_HEIGHT, 1))

def inference(inputNeurons, hiddenNeurons, inputKernels, hiddenKernels, eventInput):
    # init layers
    inputLayer = ConvLayer(2, len(hiddenKernels[0]), len(hiddenKernels[0, 0, 0]))
    convLayer = ConvLayer(CONV_CHANNELS, len(hiddenKernels[0]), len(hiddenKernels[0, 0, 0]))

    inputLayer.assignLayer(eventInput, inputKernels, inputNeurons, False)
    inputNeurons , ffQ = inputLayer.forward()
    print("%d spikes in input layer" %(ffQ.qsize()))
    num_spikes = ffQ.qsize()

    for l in range(HIDDEN_LAYERS):
        convLayer.assignLayer(ffQ, hiddenKernels[l], hiddenNeurons[l], False)
        hiddenNeurons[l], ffQ = convLayer.forward()
        print("%d spikes in layer %d" %(ffQ.qsize(), l+1))
        num_spikes += ffQ.qsize()

    return num_spikes

runs=1
time = timeit(lambda: inference(inputNeurons, hiddenNeurons, inputKernels, hiddenKernels, eventInput), number=runs)
print(f"Time: {time/runs:.6f}")

#print(" c  x  y  t")
# while outQ.qsize() > 0:
#     event = outQ.get()
#     #print("%2d %2d %2d %2d" % (event.channel, event.x_pos, event.y_pos, event.timestamp))

pass