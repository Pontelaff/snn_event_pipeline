import numpy as np
import time
from neurocore import Neurocore
from timeit import timeit
import dataloader as dl
from config import SEG_WIDTH, SEG_HEIGHT, EVENT_TIMESLICE, REC_LAYERS, NUM_INPUT
from utils import SpikeQueue, cropLogs
from visualization import compNeuronLogs, compNeuronInput, plotThresholdComp


# data type for numpy arrays
dtype = np.float16

model = dl.loadModel()
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
    spikeInput = dl.loadEventsFromArr(inPath)

    # initialise kernel weights and neuron states
    inputKernels, hiddenKernels, recKernels, outKernels = dl.loadKernelsFromModel(model)
    inputNeurons, hiddenNeurons, _ = initNeurons(len(hiddenKernels), len(inputKernels), len(hiddenKernels[0]), len(outKernels))

    num_bins = spikeInput[-1].t//EVENT_TIMESLICE + 1
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
        #recQ = loadEventsFromArr(outPath, True)
        neurons = hiddenNeurons[layerNum-1]
    else:
        neuronLogIn = np.zeros([num_bins, len(hiddenKernels[layerNum-1, 0])])
        kernels = hiddenKernels[layerNum-1]
        neurons = hiddenNeurons[layerNum-1]

    convLayer = Neurocore(len(kernels[0]), len(kernels), len(kernels[0, 0]), dtype)
    convLayer.assignLayer(spikeInput, kernels, neurons, recQ, rKernels, threshold, leak)
    neuronStates, _, ffQ, recQ = convLayer.forward(neuronLogIn, neuronLogOut, neuronLogStates, neuron)

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

def printStatsMD(spikes, checkpoints):
    print("\n| Layer\t| Spikes\t| Execution Time |\n|---|---|---|")
    for i in range(len(layerNames)-1):
        print("| %s\t| %d\t\t| %s |" %(layerNames[i], spikes[i], (checkpoints[i+2] - checkpoints[i+1])*1000))
    print("| All\t| %d\t\t| %s |" %(sum(spikes), (checkpoints[8] - checkpoints[1])*1000))

def printStats(spikes, checkpoints):
    print("\nLayer\tSpikes\tExecution Time")
    for i in range(len(layerNames)-1):
        print("%s:\t%d\t\t%s" %(layerNames[i], spikes[i], (checkpoints[i+2] - checkpoints[i+1])*1000))
    print("All:\t%d\t\t%s" %(sum(spikes), (checkpoints[8] - checkpoints[1])*1000))

def inference(logLayer, logNeuron):
    cp_time = np.zeros(9)
    cp_time[0] = time.perf_counter()
    num_spikes = np.zeros(7)
    # initialise kernel weights and neuron states
    inputKernels, hiddenKernels, recKernels, outKernels = dl.loadKernelsFromModel(model)
    inputNeurons, hiddenNeurons, _ = initNeurons(len(hiddenKernels), len(inputKernels), len(hiddenKernels[0]), len(outKernels))
    # init layers
    inputLayer = Neurocore(len(inputKernels[0]), len(hiddenKernels[0]), len(hiddenKernels[0, 0, 0]), dtype)
    convLayer = Neurocore(len(hiddenKernels[0, 0]), len(hiddenKernels[0]), len(hiddenKernels[0, 0, 0]), dtype)

    # load input events from file
    eventInput = dl.loadEvents(SEG_WIDTH, SEG_HEIGHT, NUM_INPUT)
    # load thresholds and leaks from model
    thresholds = dl.loadThresholdsFromModel(model)
    leaks = dl.loadLeakRatesFromModel(model)
    neuronLogIn = None
    neuronLogOut = None
    neuronLogStates = None


    print("Load time: %s ms" %((time.perf_counter() - cp_time[0])*1000))

    while len(eventInput) > 0:
        cp_time[1] = time.perf_counter()
        # run input layer
        inputLayer.assignLayer(eventInput, inputKernels, inputNeurons, threshold=thresholds[0], leak=leaks[0])
        inputNeurons, eventInput, ffQ, _ = inputLayer.forward()

        cp_time[2] = time.perf_counter()
        num_spikes[0] = len(ffQ)

        recQueues = [SpikeQueue() for _ in range(len(REC_LAYERS))]


        # run hidden layers
        for l in range(len(hiddenKernels)):
            try:
                recInd = REC_LAYERS.index(l+1)
                rec = True
            except ValueError as ve:
                rec = False

            if l == logLayer-1:
                num_bins = ffQ[-1].t//EVENT_TIMESLICE +1
                neuronLogOut = np.zeros([num_bins, len(hiddenKernels[l])])
                neuronLogStates = np.zeros([num_bins, len(hiddenKernels[l])])
                neuronLogIn = np.zeros([num_bins, len(hiddenKernels[l, 0])])
                ln = logNeuron
            else:
                ln = None

            if rec:
                convLayer.assignLayer(ffQ, hiddenKernels[l], hiddenNeurons[l], recQueues[recInd], recKernels[recInd], thresholds[l], leaks[l])
                hiddenNeurons[l], _, ffQ, recQueues[recInd]= convLayer.forward()
            else:
                convLayer.assignLayer(ffQ, hiddenKernels[l], hiddenNeurons[l], threshold=thresholds[l], leak=leaks[l])
                hiddenNeurons[l], _, ffQ, _ = convLayer.forward()

            num_spikes[l+1] = len(ffQ)
            cp_time[l+3] = time.perf_counter()

        np.save("test_sequences/" + layerNames[loggedLayer] + "_inLog.npy", neuronLogIn)
        np.save("test_sequences/" + layerNames[loggedLayer] + "_outLog.npy", np.stack([neuronLogStates, neuronLogOut], axis=-1))

        printStats(num_spikes, cp_time)

    print("\nInference time:\t%s ms" %((cp_time[8] - cp_time[0])*1000))

    return num_spikes

loggedLayer = 1
loggedNeuron = (18, 1, 1)
inference(loggedLayer, loggedNeuron)
# load thresholds
# thresholds = loadThresholdsFromModel(model)
# leaks = loadLeakRatesFromModel(model)
#extime = timeit(lambda: logNeuron(loggedLayer, loggedNeuron, thresholds[loggedLayer], leaks[loggedLayer]), number=1)
#print(f"Time: {extime/runs:.6f}")

#compNeuronInput(layerNames[loggedLayer])
#compNeuronLogs(layerNames[loggedLayer], loggedNeuron[0])


pass
