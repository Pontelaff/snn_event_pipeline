import numpy as np
from layers import ConvLayer
from timeit import timeit
from dataloader import loadKernels, loadEvents, loadEventsFromArr
from utils import SpikeQueue
from neurocore import LOG_NEURON, LOG_BINSIZE
from visualization import plotNeuronActivity, compNeuronLogs, compNeuronInput


MODEL_PATH = "pretrained/LIFFireNet.pth"
INPUT_PATH = "datasets/cup-drop-short.h5"
NUM_INPUT = 2000000

SEG_WIDTH = 10
SEG_HEIGHT = 10
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
#eventInput = loadEvents(INPUT_PATH, SEG_WIDTH, SEG_HEIGHT, NUM_INPUT)
eventInput = loadEventsFromArr('test_sequences/test_input_seq.npy')

if LOG_NEURON is not None:
    num_bins = eventInput[-1].t//LOG_BINSIZE +1
    logLayer = LOG_NEURON[0]
    neuronLogOut = np.zeros(num_bins)
    # TODO: different size for rec layer
    if logLayer == 0:
        neuronLogIn = np.zeros([num_bins, len(inputKernels[0])])
    else:
        neuronLogIn = np.zeros([num_bins, len(hiddenKernels[logLayer-1, 0])])
    pass
else:
    logLayer = None

def inference(inputNeurons, hiddenNeurons, inputKernels, hiddenKernels, eventInput):
    # init layers
    inputLayer = ConvLayer(len(inputKernels[0]), len(hiddenKernels[0]), len(hiddenKernels[0, 0, 0]), dtype)
    convLayer = ConvLayer(len(hiddenKernels[0, 0]), len(hiddenKernels[0]), len(hiddenKernels[0, 0, 0]), dtype)

    # run input layer
    inputLayer.assignLayer(eventInput, inputKernels, inputNeurons)
    inLog = neuronLogIn if logLayer == 0 else None
    outLog = neuronLogOut if logLayer == 0 else None
    inputNeurons, ffQ, _ = inputLayer.forward(inLog, outLog)
    print("%d spikes in input layer" %(len(ffQ)))
    num_spikes = len(ffQ)

    recQueues = [SpikeQueue() for _ in range(len(REC_LAYERS))]

    # run hidden layers
    for l in range(0): # disabled for debugging
        try:
            recInd = REC_LAYERS.index(l)
            rec = True
        except ValueError as ve:
            rec = False

        inLog = neuronLogIn if logLayer == l+1  else None
        outLog = neuronLogOut if logLayer == l+1  else None
        if rec:
            convLayer.assignLayer(ffQ, hiddenKernels[l], hiddenNeurons[l], recQueues[recInd], recKernels[recInd])
            hiddenNeurons[l], ffQ, recQueues[recInd]= convLayer.forward(inLog, outLog)
        else:
            convLayer.assignLayer(ffQ, hiddenKernels[l], hiddenNeurons[l])
            hiddenNeurons[l], ffQ, _ = convLayer.forward(inLog, outLog)

        print("%d spikes in layer %d" %(len(ffQ), l+1))
        num_spikes += len(ffQ)

    return num_spikes

runs=1
time = timeit(lambda: inference(inputNeurons, hiddenNeurons, inputKernels, hiddenKernels, eventInput), number=runs)
print(f"Time: {time/runs:.6f}")

np.save('test_sequences/neuronLogIn.npy', neuronLogIn)
np.save('test_sequences/neuronLogOut.npy', neuronLogOut)

compNeuronInput('test_sequences/test_input_seq.npy', 'test_sequences/neuronLogIn.npy')
compNeuronLogs('test_sequences/test_input_seq.npy', 'test_sequences/neuronLogIn.npy', 'test_sequences/test_output_seq.npy', 'test_sequences/neuronLogOut.npy')

#plotNeuronActivity(neuronLogIn, neuronLogOut)

#print(" c  x  y  t")
# while outQ.qsize() > 0:
#     event = outQ.get()
#     #print("%2d %2d %2d %2d" % (event.channel, event.x_pos, event.y_pos, event.timestamp))

pass