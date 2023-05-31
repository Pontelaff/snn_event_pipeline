import numpy as np
from layers import ConvLayer
from timeit import timeit
from dataloader import loadKernels, loadEvents, loadEventsFromArr
from utils import SpikeQueue
from neurocore import LOG_BINSIZE
from visualization import compNeuronLogs, compNeuronInput


MODEL_PATH = "pretrained/LIFFireNet.pth"
INPUT_PATH = "datasets/cup-drop-short.h5"
NUM_INPUT = 2000000

SEG_WIDTH = 10
SEG_HEIGHT = 10
REC_LAYERS = (1,4)

# initialise kernel weights
inputKernels, hiddenKernels, recKernels, outputKernels = loadKernels(MODEL_PATH)

# define the structured data type
dtype = np.dtype([('u', np.float16), ('t', np.int32)])
# initialise neuron states
numHiddenLayers = len(hiddenKernels)
inputNeurons = np.zeros([len(inputKernels), SEG_WIDTH, SEG_HEIGHT], dtype=dtype)
hiddenNeurons = np.zeros([numHiddenLayers, len(hiddenKernels[0]), SEG_WIDTH, SEG_HEIGHT], dtype=dtype)
outputNeurons = np.zeros([len(outputKernels), SEG_WIDTH, SEG_HEIGHT], dtype=dtype)

def testNeuronActivity(layer, channel, x_pos, y_pos):

    layerNames = ("head", "G1", "R1a", "R1b", "G2", "R2a", "R2b")

    # load input events from file
    inPath = "test_sequences/" + layerNames[layer] + "_input_seq.npy"
    spikeInput = loadEventsFromArr(inPath)

    num_bins = spikeInput[-1].t//LOG_BINSIZE + 1
    neuronLogOut = np.zeros(num_bins)
    recQ = SpikeQueue()
    rKernels = None
    if layer == 0:
        neuronLogIn = np.zeros([num_bins, len(inputKernels[0])])
        kernels = inputKernels
        neurons = inputNeurons
    elif REC_LAYERS.count(layer):
        neuronLogIn = np.zeros([num_bins, len(hiddenKernels[0])*2])
        kernels = hiddenKernels[layer-1]
        recInd = REC_LAYERS.index(layer)
        rKernels = recKernels[recInd-1]
        neurons = hiddenNeurons[layer-1]
    else:
        neuronLogIn = np.zeros([num_bins, len(hiddenKernels[layer-1, 0])])
        kernels = hiddenKernels[layer-1]
        neurons = hiddenNeurons[layer-1]

    convLayer = ConvLayer(len(kernels[0]), len(kernels), len(kernels[0, 0]), dtype)
    convLayer.assignLayer(spikeInput, kernels, neurons, recQ, rKernels)
    neuronStates, ffQ, recQ = convLayer.forward(neuronLogIn, neuronLogOut, (channel, x_pos, y_pos))
    print("%d spikes in layer %s" %(len(ffQ), layerNames[layer]))

    np.save("test_sequences/" + layerNames[layer] + "_inLog.npy", neuronLogIn)
    np.save("test_sequences/" + layerNames[layer] + "_outLog.npy", neuronLogOut)

    compNeuronInput(inPath, "test_sequences/" + layerNames[layer] + "_inLog.npy")
    compNeuronLogs(layerNames[layer], channel)

    return


def inference(inputNeurons, hiddenNeurons, inputKernels, hiddenKernels):
    # init layers
    inputLayer = ConvLayer(len(inputKernels[0]), len(hiddenKernels[0]), len(hiddenKernels[0, 0, 0]), dtype)
    convLayer = ConvLayer(len(hiddenKernels[0, 0]), len(hiddenKernels[0]), len(hiddenKernels[0, 0, 0]), dtype)

    # load input events from file
    eventInput = loadEvents(INPUT_PATH, SEG_WIDTH, SEG_HEIGHT, NUM_INPUT)

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
#time = timeit(lambda: inference(inputNeurons, hiddenNeurons, inputKernels, hiddenKernels, eventInput), number=runs)
time = timeit(lambda: testNeuronActivity(1, 18, 1, 1), number=runs)
print(f"Time: {time/runs:.6f}")


#print(" c  x  y  t")
# while outQ.qsize() > 0:
#     event = outQ.get()
#     #print("%2d %2d %2d %2d" % (event.channel, event.x_pos, event.y_pos, event.timestamp))

pass