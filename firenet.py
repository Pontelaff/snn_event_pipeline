import numpy as np
from layers import ConvLayer
from timeit import timeit
from dataloader import loadKernelsFromModel, loadThresholdsFromModel, loadModel, loadEvents, loadEventsFromArr
from utils import SpikeQueue, cropLogs
from neurocore import LOG_BINSIZE
from visualization import compNeuronLogs, compNeuronInput, plotThresholdComp


MODEL_PATH = "pretrained/LIFFireNet.pth"
INPUT_PATH = "datasets/cup-drop-short.h5"
NUM_INPUT = 2000000

SEG_WIDTH = 3
SEG_HEIGHT = 3
REC_LAYERS = (1,4)


# define the structured data type
dtype = np.dtype([('u', np.float16), ('t', np.int32)])

model = loadModel(MODEL_PATH)

def initNeurons(numHiddenLayers, numInKernels, numHiddenKernels, numOutKernels):
    # initialise neuron states
    inputNeurons = np.zeros([numInKernels, SEG_WIDTH, SEG_HEIGHT], dtype=dtype)
    hiddenNeurons = np.zeros([numHiddenLayers, numHiddenKernels, SEG_WIDTH, SEG_HEIGHT], dtype=dtype)
    outputNeurons = np.zeros([numOutKernels, SEG_WIDTH, SEG_HEIGHT], dtype=dtype)

    return inputNeurons, hiddenNeurons, outputNeurons

def logNeuron(layerNames, layerNum, neuron, threshold = None):

    # load input events from file
    inPath = "test_sequences/" + layerNames[layerNum] + "_input_seq.npy"
    outPath = "test_sequences/" + layerNames[layerNum] + "_output_seq.npy"
    spikeInput = loadEventsFromArr(inPath)

    # initialise kernel weights and neuron states
    inputKernels, hiddenKernels, recKernels, outKernels = loadKernelsFromModel(model)
    inputNeurons, hiddenNeurons, _ = initNeurons(len(hiddenKernels), len(inputKernels), len(hiddenKernels[0]), len(outKernels))

    num_bins = spikeInput[-1].t//LOG_BINSIZE + 1
    neuronLogOut = np.zeros([num_bins, len(inputKernels)])
    recQ = SpikeQueue()
    rKernels = None
    if layerNum == 0:
        neuronLogIn = np.zeros([num_bins, len(inputKernels[0])])
        kernels = inputKernels
        neurons = inputNeurons
    elif REC_LAYERS.count(layerNum):
        neuronLogIn = np.zeros([num_bins, len(hiddenKernels[0])*2])
        kernels = hiddenKernels[layerNum-1]
        recInd = REC_LAYERS.index(layerNum)
        rKernels = recKernels[recInd-1]
        recQ = loadEventsFromArr(outPath, True)
        neurons = hiddenNeurons[layerNum-1]
    else:
        neuronLogIn = np.zeros([num_bins, len(hiddenKernels[layerNum-1, 0])])
        kernels = hiddenKernels[layerNum-1]
        neurons = hiddenNeurons[layerNum-1]

    convLayer = ConvLayer(len(kernels[0]), len(kernels), len(kernels[0, 0]), dtype)
    convLayer.assignLayer(spikeInput, kernels, neurons, recQ, rKernels)
    neuronStates, ffQ, recQ = convLayer.forward(neuronLogIn, neuronLogOut, neuron, threshold)
    print("%d spikes in layer %s" %(len(ffQ), layerNames[layerNum]))

    np.save("test_sequences/" + layerNames[layerNum] + "_inLog.npy", neuronLogIn)
    np.save("test_sequences/" + layerNames[layerNum] + "_outLog.npy", neuronLogOut)

    return neuronLogOut


def testThresholds(layerNames, layerNum, neuron, thresholds):
    path = "test_sequences/" + layerNames[layerNum] + "_output_seq.npy"
    pytorchOut =  np.load(path)
    jaccard = np.zeros(len(thresholds)) # Jaccard distance
    hamming = np.zeros(len(thresholds)) # Hamming distance

    for i in range(len(thresholds)):
        ownOut = logNeuron(layerNames, layerNum, neuron, thresholds[i])
        ownOut, pytorchOut = cropLogs(ownOut, pytorchOut)
        matchingSpikes = np.logical_and(ownOut, pytorchOut)
        disjunctSpikes = (ownOut != pytorchOut)
        jaccard[i] = np.count_nonzero(disjunctSpikes)/(np.count_nonzero(ownOut + pytorchOut))
        hamming[i] = np.count_nonzero(disjunctSpikes)/(len(disjunctSpikes)*len(disjunctSpikes[0]))

    return jaccard, hamming

def inference():
    # initialise kernel weights and neuron states
    inputKernels, hiddenKernels, recKernels, outKernels = loadKernelsFromModel(model)
    inputNeurons, hiddenNeurons, _ = initNeurons(len(hiddenKernels), len(inputKernels), len(hiddenKernels[0]), len(outKernels))
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
    for l in range(len(hiddenKernels[0])):
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

layerNames =("head", "G1", "R1a", "R1b", "G2", "R2a", "R2b", "pred")
loggedLayer = 1
loggedNeuron = (18, 1, 1)
runs=1
#time = timeit(lambda: inference(), number=runs)
# load thresholds
thresholds = loadThresholdsFromModel(model)
time = timeit(lambda: logNeuron(layerNames, loggedLayer, loggedNeuron, thresholds[loggedLayer]), number=runs)
print(f"Time: {time/runs:.6f}")

compNeuronInput(layerNames[loggedLayer])
compNeuronLogs(layerNames[loggedLayer], loggedNeuron[0])

# thresholds = np.linspace(0.3, 2.0, 35)
# jac, ham = testThresholds(layerNames, loggedLayer, loggedNeuron, thresholds)
# plotThresholdComp(jac, ham, thresholds)

pass
