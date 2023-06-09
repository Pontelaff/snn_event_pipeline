import numpy as np
from layers import ConvLayer
from timeit import timeit
from dataloader import loadKernelsFromModel, loadThresholdsFromModel, loadLeakRatesFromModel, loadModel, loadEvents, loadEventsFromArr
from utils import SpikeQueue, cropLogs
from neurocore import EVENT_TIMESCLICE
from visualization import compNeuronLogs, compNeuronInput, plotThresholdComp


MODEL_PATH = "pretrained/LIFFireNet.pth"
INPUT_PATH = "datasets/cup-drop-short.h5"
NUM_INPUT = 2000000

SEG_WIDTH = 16
SEG_HEIGHT = 16
REC_LAYERS = (1,4)


# define the structured data type
dtype = np.float16

model = loadModel(MODEL_PATH)
layerNames =("head", "G1", "R1a", "R1b", "G2", "R2a", "R2b", "pred")

def initNeurons(numHiddenLayers, numInKernels, numHiddenKernels, numOutKernels):
    # initialise neuron states
    inputNeurons = np.zeros([numInKernels, SEG_WIDTH, SEG_HEIGHT], dtype=dtype)
    hiddenNeurons = np.zeros([numHiddenLayers, numHiddenKernels, SEG_WIDTH, SEG_HEIGHT], dtype=dtype)
    outputNeurons = np.zeros([numOutKernels, SEG_WIDTH, SEG_HEIGHT], dtype=dtype)

    return inputNeurons, hiddenNeurons, outputNeurons

def logNeuron(layerNum, neuron, threshold = None, leak = None):

    # load input events from file
    inPath = "test_sequences/" + layerNames[layerNum] + "_input_seq.npy"
    outPath = "test_sequences/" + layerNames[layerNum] + "_output_seq.npy"
    spikeInput = loadEventsFromArr(inPath)

    # initialise kernel weights and neuron states
    inputKernels, hiddenKernels, recKernels, outKernels = loadKernelsFromModel(model)
    inputNeurons, hiddenNeurons, _ = initNeurons(len(hiddenKernels), len(inputKernels), len(hiddenKernels[0]), len(outKernels))

    num_bins = spikeInput[-1].t//EVENT_TIMESCLICE + 1
    neuronLogOut = np.zeros([num_bins, len(inputKernels)])
    neuronLogStates = np.zeros([num_bins, len(inputKernels)])
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
        rKernels = recKernels[recInd]
        recQ = loadEventsFromArr(outPath, True)
        neurons = hiddenNeurons[layerNum-1]
    else:
        neuronLogIn = np.zeros([num_bins, len(hiddenKernels[layerNum-1, 0])])
        kernels = hiddenKernels[layerNum-1]
        neurons = hiddenNeurons[layerNum-1]

    convLayer = ConvLayer(len(kernels[0]), len(kernels), len(kernels[0, 0]), dtype)
    convLayer.assignLayer(spikeInput, kernels, neurons, recQ, rKernels, threshold, leak)
    neuronStates, ffQ, recQ = convLayer.forward(neuronLogIn, neuronLogOut, neuronLogStates, neuron)
    print("%d spikes in layer %s" %(len(ffQ), layerNames[layerNum]))

    np.save("test_sequences/" + layerNames[layerNum] + "_inLog.npy", neuronLogIn)
    np.save("test_sequences/" + layerNames[layerNum] + "_outLog.npy", np.stack([neuronLogStates, neuronLogOut], axis=-1))

    return neuronLogOut

def testThresholds(layerNum, neuron, thresholds):
    path = "test_sequences/" + layerNames[layerNum] + "_output_seq.npy"
    pytorchOut =  np.load(path)[:,:,1,1]
    jaccard = np.zeros(len(thresholds)) # Jaccard distance
    hamming = np.zeros(len(thresholds)) # Hamming distance

    for i in range(len(thresholds)):
        ownOut = logNeuron(layerNames, layerNum, neuron, thresholds[i])
        ownOut, pytorchOut = cropLogs(ownOut[:,:,1], pytorchOut)
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
    # load thresholds and leaks from model
    thresholds = loadThresholdsFromModel(model)
    leaks = loadLeakRatesFromModel(model)

    # run input layer
    inputLayer.assignLayer(eventInput, inputKernels, inputNeurons, threshold=thresholds[0], leak=leaks[0])
    inputNeurons, ffQ, _ = inputLayer.forward()
    print("%d spikes in input layer" %(len(ffQ)))
    num_spikes = len(ffQ)

    recQueues = [SpikeQueue() for _ in range(len(REC_LAYERS))]

    # run hidden layers
    for l in range(len(hiddenKernels)):
        try:
            recInd = REC_LAYERS.index(l+1)
            rec = True
        except ValueError as ve:
            rec = False

        if rec:
            convLayer.assignLayer(ffQ, hiddenKernels[l], hiddenNeurons[l], recQueues[recInd], recKernels[recInd], thresholds[l], leaks[l])
            hiddenNeurons[l], ffQ, recQueues[recInd]= convLayer.forward()
        else:
            convLayer.assignLayer(ffQ, hiddenKernels[l], hiddenNeurons[l], threshold=thresholds[l], leak=leaks[l])
            hiddenNeurons[l], ffQ, _ = convLayer.forward()

        print("%d spikes in layer %s" %(len(ffQ), layerNames[l+1]))
        num_spikes += len(ffQ)

    return num_spikes

loggedLayer = 2
loggedNeuron = (18, 1, 1)
runs=1
#time = timeit(lambda: inference(), number=runs)
# load thresholds
thresholds = loadThresholdsFromModel(model)
leaks = loadLeakRatesFromModel(model)
time = timeit(lambda: logNeuron(layerNames, loggedLayer, loggedNeuron, thresholds[loggedLayer], leaks[loggedLayer]), number=runs)
print(f"Time: {time/runs:.6f}")

compNeuronInput(layerNames[loggedLayer])
compNeuronLogs(layerNames[loggedLayer], loggedNeuron[0])

# thresholds = np.linspace(0.8, 1.3, 21)
# jac, ham = testThresholds(layerNames, loggedLayer, loggedNeuron, thresholds)
# plotThresholdComp(jac, ham, thresholds)

pass
