import os
import torch
import h5py
import numpy as np
from typing import Tuple
from models.model import FireNet, LIFFireNet
from utils import SpikeQueue, Spike


def loadKernels(modelPath) -> Tuple:
    """
    The function loads kernels from a saved PyTorch model and returns them as a tuple.

    @param modelPath The path to the saved PyTorch model file.

    @return A tuple containing four arrays of weights: inputKernels, hiddenKernels, recKernels, and
    outputKernels.
    """
    if os.path.isfile(modelPath):
        model = torch.load(modelPath, 'cpu')
        print("Model restored from " + modelPath + "\n")
    else:
        print("No model found at" + modelPath + "\n")
        return None

    # extract kernels
    inputKernels = model.head.ff.weight.detach().numpy()

    hiddenLayers = (model.G1, model.R1a, model.R1b, model.G2, model.R2a, model.R2b)
    hiddenKernels = np.array([hiddenLayers[l].ff.weight.detach().numpy() for l in range(len(hiddenLayers))])
    recKernels = np.array([model.G1.rec.weight.detach().numpy(), model.G2.rec.weight.detach().numpy()])

    outputKernels = model.pred.conv2d.weight.detach().numpy()

    return (inputKernels, hiddenKernels, recKernels, outputKernels)

def loadEvents(filePath, numEvents = -1) -> SpikeQueue:
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
        mask = (events[:, 0] >= 300) & (events[:, 0] <= 331) & (events[:, 1] <= 31) &\
            (events[:,3] >= 80000) & (events[:,3] <= 180000)
        ev = events[mask]

        # subtract offset to start at t = 0
        t_offset = ev[0,3]
        ev[:,3] -= t_offset

        spikeQueue = [Spike(ev[i][0]-300, ev[i][1], int(ev[i][2]), ev[i][3]) for i in range(len(ev))]
        print(f"{len(spikeQueue)} input events read from {filePath}\n")
    else:
        print("File not found at " + filePath + "\n")
        return None

    return spikeQueue