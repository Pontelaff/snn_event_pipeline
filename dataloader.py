import os
import torch
from models.model import FireNet, LIFFireNet
import numpy as np



def loadKernels(model_dir, device):
    if os.path.isfile(model_dir):
        model = torch.load(model_dir, map_location=device)
        print("Model restored from " + model_dir + "\n")
    else:
        print("No model found at" + model_dir + "\n")

    inputKernels = model.head.ff.weight.detach().numpy()

    hiddenLayers = (model.G1, model.R1a, model.R1b, model.G2, model.R2a, model.R2b)
    hiddenKernels = np.array([hiddenLayers[l].ff.weight.detach().numpy() for l in range(len(hiddenLayers))])
    recKernels = np.array([model.G1.rec.weight.detach().numpy(), model.G2.rec.weight.detach().numpy()])

    outputKernels = model.pred.conv2d.weight.detach().numpy()

    return (inputKernels, hiddenKernels, recKernels, outputKernels)