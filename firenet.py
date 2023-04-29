import numpy as np
from utils import Event, EventQueue
from layers import ConvLayer


KERNEL_SIZE = 3
CONV_CHANNELS = 32
HIDDEN_LAYERS = 6
INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 2
SEG_WIDTH = 32
SEG_HEIGHT = 32
INPUT_EVENTS = 1000

# initialise neuron states
inputNeurons = np.zeros([CONV_CHANNELS, SEG_WIDTH, SEG_HEIGHT, 2], dtype=np.float16)
hiddenNeurons = np.zeros([6, CONV_CHANNELS, SEG_WIDTH, SEG_HEIGHT, 2], dtype=np.float16)
outputNeurons = np.zeros([SEG_WIDTH, SEG_HEIGHT, 2], dtype=np.float16)

# initialise kernel weights
inputKernels = np.random.rand(CONV_CHANNELS, INPUT_CHANNELS, KERNEL_SIZE, KERNEL_SIZE).astype(np.float16)
hiddenKernels = np.random.rand(6, CONV_CHANNELS, CONV_CHANNELS, KERNEL_SIZE, KERNEL_SIZE).astype(np.float16)
outputKernels = np.random.rand(OUTPUT_CHANNELS, CONV_CHANNELS, CONV_CHANNELS, 1, 1).astype(np.float16)

# initialise event queues
eventInput = EventQueue()

x = np.random.randint(0, SEG_WIDTH, INPUT_EVENTS)
y = np.random.randint(0, SEG_HEIGHT, INPUT_EVENTS)
c = np.random.randint(0, 2, INPUT_EVENTS) # polarity
t = 0

for i in range(INPUT_EVENTS):
    t += np.random.rand()*0.25
    eventInput.put(Event(x[i], y[i], t//1, c[i]))

# recognizable test values
#hiddenKernels[0,0,0,0] = [1,2,3]
#hiddenKernels[0,1,0,0] = [44,55,66]
#hiddenKernels[0,2,0,1] = [-55,55,-55]
#hiddenNeurons[0,0,11,23:26] = [[0,1],[2,3],[55,10]]

# init layers
inputLayer = ConvLayer(2, len(hiddenKernels[0]), len(hiddenKernels[0, 0, 0]))
convLayer = ConvLayer(CONV_CHANNELS, len(hiddenKernels[0]), len(hiddenKernels[0, 0, 0]))


inputLayer.assignLayer(eventInput, inputKernels, inputNeurons, False)
inputNeurons , ffQ = inputLayer.forward(None)
print("%d spikes in input layer" %(ffQ.qsize()))

for l in range(HIDDEN_LAYERS):
    convLayer.assignLayer(ffQ, hiddenKernels[l], hiddenNeurons[l], False)
    hiddenNeurons[l], ffQ = convLayer.forward(None)
    print("%d spikes in layer %d" %(ffQ.qsize(), l+1))

#print(" c  x  y  t")
# while outQ.qsize() > 0:
#     event = outQ.get()
#     #print("%2d %2d %2d %2d" % (event.channel, event.x_pos, event.y_pos, event.timestamp))

pass