import os
import torch
import h5py
import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike
from models.model import FireNet, LIFFireNet
from utils import SpikeQueue, Spike

FRAME_OFFSET = (240, 10)
TIME_START = 80000
TIME_STOP = 180000

def sigmoid(x):
 return 1/(1 + np.exp(-x))

def loadModel(modelPath) -> LIFFireNet:
    if os.path.isfile(modelPath):
        model = torch.load(modelPath, 'cpu')
        print("Model restored from " + modelPath + "\n")
    else:
        print("No model found at" + modelPath + "\n")
        return None

    return model

def loadThresholdsFromModel(model) -> ArrayLike:
    layers = (model.head, model.G1, model.R1a, model.R1b, model.G2, model.R2a, model.R2b)
    thresholds = np.array([layers[l].thresh.detach().numpy() for l in range(len(layers))])

    return thresholds

def loadLeakRatesFromModel(model) -> ArrayLike:
    layers = (model.head, model.G1, model.R1a, model.R1b, model.G2, model.R2a, model.R2b)
    leaks = np.array([layers[l].leak.detach().numpy() for l in range(len(layers))])
    leaks = sigmoid(leaks)

    return leaks

def loadKernelsFromModel(model) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """
    The function loads kernels from a saved PyTorch model and returns them as a tuple of arrays.

    @param model The pytorch model to load the kernel weights from.

    @return A tuple containing four arrays of weights: inputKernels, hiddenKernels, recKernels, and
    outputKernels.
    """

    # extract kernels
    # Kernels are saved in model as [N x C x H x W], this implementation uses [N x C x W x H]
    # Therefore the x and y axis first need to be swapped
    # flipping is necessary as kernels are applied from previous to current layer which means
    # they need to be mirrored on both x and y axis.
    inputKernels = np.flip(np.swapaxes(model.head.ff.weight.detach().numpy(), -1, -2), axis=(-1,-2))

    hiddenLayers = (model.G1, model.R1a, model.R1b, model.G2, model.R2a, model.R2b)
    hiddenKernels = np.flip(np.swapaxes(np.array([hiddenLayers[l].ff.weight.detach().numpy() for l in range(len(hiddenLayers))]), -1, -2), axis=(-1,-2))
    recKernels = np.flip(np.swapaxes(np.array([model.G1.rec.weight.detach().numpy(), model.G2.rec.weight.detach().numpy()]), -1, -2), axis=(-1,-2))

    outputKernels = np.flip(np.swapaxes(model.pred.conv2d.weight.detach().numpy(), -1, -2), axis=(-1,-2))

    return (inputKernels, hiddenKernels, recKernels, outputKernels)

def loadEventsFromArr(arrPath, recurrent = False) -> SpikeQueue:
    """
    The function loads events from a numpy array and creates a SpikeQueue object containing the events.

    @param arrPath The path to a numpy array file containing event data.

    @return A list of Spike tuples, which is assigned to the variable `spikeQueue`.
    """
    eventArr = np.load(arrPath)
    eventIdx = np.where(eventArr >= 1)

    spikeQueue = []
    for t, c, y, x in zip(*eventIdx):
        spikeNum = int(eventArr[t, c, y, x])
        for _ in range(spikeNum):
            spikeQueue.append(Spike(x, y, c,(t+int(recurrent))*100))

    print(f"{len(spikeQueue)} input events read from {arrPath}\n")

    return spikeQueue

def loadEvents(filePath, frameWidth, frameHeight, numEvents = -1) -> SpikeQueue:
    """
    This function loads events from a specified file path, filters them based on certain criteria,
    subtracts an offset to start at t=0, and returns a list of Spike objects.

    @param filePath The file path of the input file containing events data.
    @param numEvents The number of events to load from the file. If set to -1, it will load all events
    in the file.

    @return A Spike Queue as a list of Spike tuples
    """
    if os.path.isfile(filePath):
        file = h5py.File(filePath, 'r')
        if numEvents == -1:
            numEvents = file.attrs["num_events"]
        events = np.zeros((numEvents, 4), dtype=np.int32)
        events[:,0] = file["events/xs"][:numEvents]
        events[:,1] = file["events/ys"][:numEvents]
        events[:,2] = file["events/ps"][:numEvents]
        events[:,3] = file["events/ts"][:numEvents]
        file.close()

        # filter a 32x32 window of the dropping cup
        mask = (events[:, 0] >= FRAME_OFFSET[0]) & (events[:, 0] <= FRAME_OFFSET[0] + frameWidth-1) &\
            (events[:, 1] >= FRAME_OFFSET[1]) & (events[:, 1] <= FRAME_OFFSET[1] + frameHeight-1) &\
            (events[:, 3] >= TIME_START) & (events[:,3] <= TIME_STOP)
        ev = events[mask]

        # subtract offset to start at t = 0
        t_offset = ev[0,3]
        ev[:,3] -= t_offset

        spikeQueue = [Spike(ev[i][0]-FRAME_OFFSET[0], ev[i][1]-FRAME_OFFSET[1], int(ev[i][2]), ev[i][3]) for i in range(len(ev))]
        print(f"{len(spikeQueue)} input events read from {filePath}\n")
    else:
        print("File not found at " + filePath + "\n")
        return None

    return spikeQueue