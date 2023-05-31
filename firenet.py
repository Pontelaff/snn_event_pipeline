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

def testLayer(layer):

    layerNames = ("head", "G1", "R1a", "R1b", "G2", "R2a", "R2b")

    # load input events from file
    inPath = "test_sequences/" + layerNames[layer] + "_input_seq.npy"
    spikeInput = loadEventsFromArr(inPath)


    if LOG_NEURON is not None:
        num_bins = spikeInput[-1].t//LOG_BINSIZE + 1
        logLayer = LOG_NEURON[0]
        neuronLogOut = np.zeros(num_bins)
        recQ = SpikeQueue()
        rKernels = None
        if logLayer == 0:
            neuronLogIn = np.zeros([num_bins, len(inputKernels[0])])
            kernels = inputKernels
            neurons = inputNeurons
        elif REC_LAYERS.count(logLayer-1):
            neuronLogIn = np.zeros([num_bins, len(hiddenKernels[0])*2])
            kernels = hiddenKernels[logLayer-1]
            recInd = REC_LAYERS.index(logLayer-1)
            rKernels = recKernels[recInd]
            neurons = hiddenNeurons[logLayer-1]
        else:
            neuronLogIn = np.zeros([num_bins, len(hiddenKernels[logLayer-1, 0])])
            kernels = hiddenKernels[logLayer-1]
            neurons = hiddenNeurons[logLayer-1]
        pass
    else:
        print("No neuron selected for logging\n")
        return

    convLayer = ConvLayer(len(kernels[0]), len(kernels), len(kernels[0, 0]), dtype)
    convLayer.assignLayer(spikeInput, kernels, neurons, recQ, rKernels)
    neuronStates, ffQ, recQ = convLayer.forward(neuronLogIn, neuronLogOut)
    print("%d spikes in layer %s" %(len(ffQ), layerNames[layer]))

    np.save("test_sequences/" + layerNames[layer] + "_inLog.npy", neuronLogIn)
    np.save("test_sequences/" + layerNames[layer] + "_outLog.npy", neuronLogOut)

    compNeuronInput(inPath, "test_sequences/" + layerNames[layer] + "_inLog.npy")
    compNeuronLogs(layerNames[layer])

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
time = timeit(lambda: testLayer(1), number=runs)
print(f"Time: {time/runs:.6f}")

#plotNeuronActivity(neuronLogIn, neuronLogOut)

#print(" c  x  y  t")
# while outQ.qsize() > 0:
#     event = outQ.get()
#     #print("%2d %2d %2d %2d" % (event.channel, event.x_pos, event.y_pos, event.timestamp))

pass