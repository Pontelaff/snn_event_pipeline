import os
import torch
import h5py
import numpy as np
from typing import Tuple
from models.model import FireNet, LIFFireNet
from utils import SpikeQueue, Spike


def loadKernels(modelPath, device) -> Tuple:
    if os.path.isfile(modelPath):
        model = torch.load(modelPath, map_location=device)
        print("Model restored from " + modelPath + "\n")
    else:
        print("No model found at" + modelPath + "\n")
        return None

    inputKernels = model.head.ff.weight.detach().numpy()

    hiddenLayers = (model.G1, model.R1a, model.R1b, model.G2, model.R2a, model.R2b)
    hiddenKernels = np.array([hiddenLayers[l].ff.weight.detach().numpy() for l in range(len(hiddenLayers))])
    recKernels = np.array([model.G1.rec.weight.detach().numpy(), model.G2.rec.weight.detach().numpy()])

    outputKernels = model.pred.conv2d.weight.detach().numpy()

    return (inputKernels, hiddenKernels, recKernels, outputKernels)

def loadEvents(filePath, numEvents) -> SpikeQueue:
    if os.path.isfile(filePath):
        file = h5py.File(filePath, 'r')
        xs = file["events/xs"][:numEvents]
        ys = file["events/ys"][:numEvents]
        ts = file["events/ts"][:numEvents]
        ps = file["events/ps"][:numEvents]
        spikeQueue = [Spike(xs[i], ys[i], int(ps[i]), ts[i]) for i in range(len(xs))]
        print("Input events read from " + filePath + "\n")
    else:
        print("File not found at " + filePath + "\n")
        return None


    return spikeQueue