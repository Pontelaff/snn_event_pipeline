import numpy as np
from utils import Event, EventQueue
from layers import ConvLayer


KERNEL_SIZE = 3
CONV_CHANNELS = 32
INPUT_CHANNELS = 2
OUTPUT_CHANNELS = 2
SEG_WIDTH = 32
SEG_HEIGHT = 32
INPUT_EVENTS = 32


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

x = np.random.randint(0, SEG_WIDTH, INPUT_EVENTS)
y = np.random.randint(0, SEG_HEIGHT, INPUT_EVENTS)
c = np.random.randint(0, 31, INPUT_EVENTS)
t = 0

for i in range(INPUT_EVENTS):
    t += np.random.rand()*0.25
    eventInput.put(Event(x[i], y[i], t//1, c[i]))

#inputLayer = ConvLayer(INPUT_CHANNELS, len(inputKernels), len(inputKernels[0]))
convLayer = ConvLayer(CONV_CHANNELS, len(hiddenKernels[0]), len(hiddenKernels[0, 0, 0]))


# recognizable test values
hiddenKernels[0,0,0,0] = [1,2,3]
hiddenKernels[0,1,0,0] = [44,55,66]
hiddenKernels[0,2,0,1] = [-55,55,-55]
hiddenNeurons[0,0,11,23:26] = [[0,1],[2,3],[55,10]]

convLayer.assignLayer(1, hiddenKernels[0], False, hiddenNeurons[0])
outQ = convLayer.forward(eventInput)

print("%d spikes in queue" %(outQ.qsize()))
#print(" c  x  y  t")
while outQ.qsize() > 0:
    event = outQ.get()
    #print("%2d %2d %2d %2d" % (event.channel, event.x_pos, event.y_pos, event.timestamp))

pass