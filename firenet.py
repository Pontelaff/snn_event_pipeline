import numpy as np
import time
from neurocore import Neurocore
import dataloader as dl
from config import SEG_WIDTH, SEG_HEIGHT, REC_LAYERS, NUM_INPUT
from utils import SpikeQueue



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

def inference():
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

            # Use recurrent kernels and spike queues for recurrent layers
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

inference()
