import numpy as np
from utils import Spike, SpikeQueue
from typing import Tuple
from numpy.typing import ArrayLike

EVENT_TIMESLICE = 1000
REFRACTORY_PERIOD = 50

def areNeighbours(x_off, y_off, kernelSize) -> bool:
    """
    The function checks if two neurons with a given offset to each other are neighbours within a given
    kernel size and thus affected by the same kernel.

    @param x_off The horizontal offset between the neurons.
    @param y_off The vertical offset between the neurons.
    @param kernelSize The size of the square kernel. It is used to determine the maximum distance between
    two neurons for them to be considered neighbors.

    @return A boolean value indicating whether the neurons with distance `x_off, y_off` are neigbours
    based on the given kernel size.
    """
    neighbours = max(abs(x_off), abs(y_off)) <= kernelSize//2
    return neighbours

class Neurocore:
    # member attributes
    kernels = None      # 32*3*3 numpy array containing one channel each of 32 Kernels
    recKernels = None   # recurrent Kernels
    spike = None    # spike for convolution step

    def __init__(self, channel, numKernels, kernelSize, dtype) -> None:
        """
        This is the initialization function for a neurocore managing input from one channel in a convolutional
        neural network layer.

        @param channel The number of the channel, which this neurocore is receiving spikes from.
        @param numKernels The number of kernels in the layer. This also determines the number of output channels.
        @param kernelSize The height and with of the kernels
        @param dtype datatype of the neuron states
        """
        self.channel = channel
        self.kernelSize = kernelSize
        self.neuronStates = np.zeros([numKernels,kernelSize,kernelSize], dtype=dtype) # numpy array containing neighbours of spiking neuron

    def assignLayer(self, kernels, recKernels = None):
        """
        This function assigns a new layer and loads kernels for that layer based on the channel
        specified for the Neurocore.

        @param kernels All kernals of the neural network layer as a numpy array
                [Kernals, Channels, KSize, KSize]
        @param recKernels All recurrent kernels. None, if layer is not recurrent.
        """
        # from active layer for all kernels load the designated channel
        self.kernels = kernels[:, self.channel]
        if recKernels is not None:
            self.recKernels = recKernels[:, self.channel]

    def loadNeurons(self, s: Spike, neurons):
        """
        This function loads neuron states neighbouring spike coordinates for each channel of the current
        layer.

        @param s A Spike object that contains information about the location of a spike in the neural
        network.
        @param neurons A 3-dimensional numpy array representing the neurons in a layer. The first
        dimension represents the channel, the second and third dimensions represent the x and y
        positions of the neurons in the layer.
        """

        # pad each channel with zeros (don't pad neuron states)
        neurons = np.pad(neurons, ((0,0),(1,1),(1,1)), 'constant')
        # for each channel of current layer
        # load neuron states neighbouring spike coordinates
        # start pos-1 stop pos+2 and increment by 1 to account for padding
        self.neuronStates = neurons[:, s.x:s.x+3, s.y:s.y+3]
        self.spike = s

    def applyConv(self, recSpike, neuronInLog = None, ln = None) -> ArrayLike:
        """
        This function performs the convolution operations for neurons neighbouring the current spike
        and one channel (specified by the neurocore) of each kernel. Each kernel will then apply the
        respective weights to a different channel of the current layer.

        @param recSpike A boolean parameter that indicates whether the convolution operation is being
        performed for a recurrent spike or not. If it is True, then the recurrent kernels will
        be used instead of the regular kernels.
        @param neuronInLog A numpy array used to log all weighted input spikes seperated by channel
        and time bins for the observed neuron.
        @param neuronOutLog A numpy array used to log all output spikes seperated by time bins for the
        observed neuron.
        @param ln The `ln` parameter is an optional argument that specifies a single neuron whose
        activity should be logged. If this parameter is not provided, no neuron activity will be logged.

        @return The updated neuron states array after performing the convolution operation.
        """
        kernels = self.recKernels if recSpike else self.kernels
        self.neuronStates += kernels

        # Log weighted neuron input
        if ln is not None:
            x_offset =  ln[1] -self.spike.x
            y_offset = ln[2] - self.spike.y
            if areNeighbours(x_offset, y_offset, self.kernelSize):
                bin = self.spike.t//EVENT_TIMESLICE
                neuronInLog[bin, self.spike.c + int(recSpike) * 32] += kernels[ln[0], x_offset + 1, y_offset+1]

    def forward(self, s: Spike, neurons, recSpike, neuronInLog = None, loggedNeuron = None)\
                -> Tuple[ArrayLike, SpikeQueue, SpikeQueue]:
        """
        This function performs forward propagation in a neural network by loading neurons and
        performing convolution.

        @param s A named tuple containing coordinates and timestamp of the spike to be processed.
        @param neurons A numpy array containing the state of each neuron in the layer.
        @param recSpike A boolean parameter that indicates whether the convolution operation is being
        performed for a recurrent spike or not. If it is True, then the recurrent kernels will
        be used instead of the regular kernels.
        @param neuronInLog A numpy array used to log all weighted input spikes seperated by channel
        and time bins for the observed neuron.
        @param neuronOutLog A numpy array used to log all output spikes seperated by time bins for the
        observed neuron.
        @param loggedNeuron The `loggedNeuron` parameter is an optional argument that specifies a single
        neuron whose activity should be logged. If this parameter is not provided, no neuron activity
        will be logged.

        @return A numpy array containing the updated neuron states.
        """
        # load neurons into neurocore
        self.loadNeurons(s, neurons)

        # perform convolution and generate spikes
        self.applyConv(recSpike, neuronInLog, loggedNeuron)


        return self.neuronStates
