import numpy as np
import time
from neurocore import Neurocore
import dataloader as dl
from config import *
from utils import SpikeQueue


def initNeurons(numHiddenLayers, numInKernels, numHiddenKernels, numOutKernels):
    # Initialise neuron states
    inputNeurons = np.zeros([numInKernels, SEG_WIDTH, SEG_HEIGHT], dtype=DATATYPE)
    hiddenNeurons = np.zeros([numHiddenLayers, numHiddenKernels, SEG_WIDTH, SEG_HEIGHT], dtype=DATATYPE)
    outputNeurons = np.zeros([numOutKernels, SEG_WIDTH, SEG_HEIGHT], dtype=DATATYPE)

    return inputNeurons, hiddenNeurons, outputNeurons

def printLayerStatsMD(spikes, checkpoints):
    print("\n| Layer\t| Spikes\t| Execution Time |\n|---|---|---|")
    for i in range(len(FIRENET_LAYERS)-1):
        print("| %s\t| %d\t\t| %s |" %(FIRENET_LAYERS[i], spikes[i], (checkpoints[i+1] - checkpoints[i])*1000))
    print("| All\t| %d\t\t| %s |" %(sum(spikes), (checkpoints[-1] - checkpoints[0])*1000))

def printLayerStats(spikes, checkpoints):
    print("\nLayer\tSpikes\tExecution Time")
    for i in range(len(FIRENET_LAYERS)-1):
        print("%s:\t%d\t\t%s" %(FIRENET_LAYERS[i], spikes[i], (checkpoints[i+1] - checkpoints[i])*1000))
    print("All:\t%d\t\t%s" %(sum(spikes), (checkpoints[-1] - checkpoints[0])*1000))

def inference():
    # Checkpoint timestamps for each layer completion
    startTime = time.perf_counter()
    # Initialise kernel weights and neuron states
    inputKernels, hiddenKernels, recKernels, outKernels = dl.loadKernelsFromModel(model)
    inputNeurons, hiddenNeurons, _ = initNeurons(len(hiddenKernels), len(inputKernels), len(hiddenKernels[0]), len(outKernels))
    # Initialise layers, inputLayer with 2 input channels and convLayer with 32 input channels
    # convLayer is reused by all convolutional layers to model the behaviour of the later hardware implementation
    inputLayer = Neurocore(len(inputKernels[0]), len(hiddenKernels[0]), len(hiddenKernels[0, 0, 0]), DATATYPE)
    convLayer = Neurocore(len(hiddenKernels[0, 0]), len(hiddenKernels[0]), len(hiddenKernels[0, 0, 0]), DATATYPE)

    # Load input events from file
    eventInput = dl.loadEvents(SEG_WIDTH, SEG_HEIGHT, NUM_INPUT)
    # Load thresholds and leaks from model
    thresholds = dl.loadThresholdsFromModel(model)
    leaks = dl.loadLeakRatesFromModel(model)


    print("Load time: %s ms" %((time.perf_counter() - startTime)*1000))

    while len(eventInput) > 0:
        # Generated spikes per layer
        spikesPerLayer = []
        # Checkpoint timestamps for each layer
        checkpointTimes = []

        # Run inference on input layer
        checkpointTimes.append(time.perf_counter())
        inputLayer.assignLayer(eventInput, inputKernels, inputNeurons, threshold=thresholds[0], leak=leaks[0])
        inputNeurons, eventInput, forwardSpikes, _ = inputLayer.forward()


        # Save inference stats for input layer
        checkpointTimes.append(time.perf_counter())
        spikesPerLayer.append(len(forwardSpikes))

        # Create a recurrent spike queue for each of the recurrent layers
        recSpikes = [SpikeQueue() for _ in range(len(REC_LAYERS))]

        # Run hidden layers
        for layerIdx in range(len(hiddenKernels)):
            # Check if layer is recurrent
            try:
                recLayerIdx = REC_LAYERS.index(layerIdx+1)
                isRecurrent = True
            except ValueError as ve:
                isRecurrent = False

            if isRecurrent:
                # Use recurrent kernels and spike queues for recurrent layers
                convLayer.assignLayer(forwardSpikes, hiddenKernels[layerIdx], hiddenNeurons[layerIdx], recSpikes[recLayerIdx], recKernels[recLayerIdx], thresholds[layerIdx], leaks[layerIdx])
                hiddenNeurons[layerIdx], _, forwardSpikes, recSpikes[recLayerIdx]= convLayer.forward()
            else:
                convLayer.assignLayer(forwardSpikes, hiddenKernels[layerIdx], hiddenNeurons[layerIdx], threshold=thresholds[layerIdx], leak=leaks[layerIdx])
                hiddenNeurons[layerIdx], _, forwardSpikes, _ = convLayer.forward()

            # Save inference stats for hidden layer
            spikesPerLayer.append(len(forwardSpikes))
            checkpointTimes.append(time.perf_counter())

        printLayerStats(spikesPerLayer, checkpointTimes)

    print("\nInference time:\t%s ms" %((checkpointTimes[-1] - startTime)*1000))

# Load the LIFFireNet PyTorch model
model = dl.loadModel()

inference()
