import numpy as np
from utils import Spike, Event
from typing import List, Tuple
from numpy.typing import ArrayLike

LEAK_RATE = 0.17
U_RESET = 0
U_THRESH = 1


def applyLeak(u, t_last, t_now) -> Tuple:
    """
    This function applies a leak to a neuron based on the time elapsed since the last
    application of the leak.

    @param u The membrane potential of the neuron that is being modified by the leak rate.
    @param t_last The timestamp of the last update for this neuron.
    @param t_now The timestamp of the incoming spike.

    @return A tuple containing the updated neuron state.

    NOTE: This might be optimised further
    TODO: Timestamp update might need to be done at threshold check.
    """
    t_leak = t_now - t_last
    # leak neuron if timestamps are different and potential is not zero
    if t_leak*u != 0:
        leak = LEAK_RATE / t_leak
        u = u*leak
        t_last = t_now

    return (u, t_last)


class Neurocore:

    # member attributes
    kernels = None      # 32*3*3 numpy array containing one channel each of 32 Kernels
    spikeLeak = None    # spike for neuron Leak step
    spikeConv = None    # spike for convolution step

    def __init__(self, channel, numKernels, kernelSize) -> None:
        """
        This is the initialization function for a neurocore managing input from one channel in a convolutional
        neural network layer.

        @param channel The number of the channel, which this neurocore is receiving spikes from.
        @param numKernels The number of kernels in the layer. This also determines the number of output channels.
        @param kernelSize The height and with of the kernels
        """
        self.channel = channel
        self.kernelSize = kernelSize
        self.neuronStatesLeak = np.zeros([numKernels,kernelSize,kernelSize,2], dtype=np.float16) # numpy array containing neighbours of spiking neuron
        self.neuronStatesConv = self.neuronStatesLeak

    def assignLayer(self, kernels):
        """
        This function assigns a new layer and loads kernels for that layer based on the channel
        specified for the Neurocore.

        @param kernels All kernals of the neural network layer as a numpy array
                [Kernals, Channels, KSize, KSize]
        """
        # from active layer for all kernels load the designated channel
        self.kernels = kernels[:, self.channel]

    def loadNeurons(self, s: Spike, neurons):
        """
        This function loads neuron states neighbouring spike coordinates for each channel of the current
        layer.

        @param s A Spike object that contains information about the location of a spike in the neural
        network.
        @param neurons A 4-dimensional numpy array representing the neurons in a layer. The first
        dimension represents the channel, the second and third dimensions represent the x and y
        positions of the neurons in the layer, the forth dimension contains the neuron states.
        """

        # pad each channel with zeros (don't pad neuron states)
        neurons = np.pad(neurons, ((0,0),(1,1),(1,1),(0,0)), 'constant')
        channels = len(neurons)
        # for each channel of current layer
        # load neuron states neighbouring spike coordinates
        # start pos-1 stop pos+2 and increment by 1 to account for padding
        self.neuronStatesLeak = neurons[:, s.x_pos:s.x_pos+3, s.y_pos:s.y_pos+3]
        self.spikeLeak = s

    def leakNeurons(self):
        """
        This function applies a leak to the neuron states and forwards the spike object to the next
        pipline step performing the convolution.
        """
        leakFunc = np.vectorize(applyLeak, otypes=[np.float16, np.int16])
        u = self.neuronStatesLeak[:,:,:,0]
        t = self.neuronStatesLeak[:,:,:,1]
        updated_neurons = leakFunc(u, t, self.spikeLeak.timestamp)
        self.neuronStatesLeak[:,:,:,0] = updated_neurons[0]
        self.neuronStatesLeak[:,:,:,1] = updated_neurons[1]
        # forward spike and neurons to convolution step
        self.spikeConv = self.spikeLeak
        self.neuronStatesConv = self.neuronStatesLeak

    def applyConv(self, recurrent = False) -> ArrayLike:
        """
        This function performs the convolution operations for neurons neighbouring the current spike
        and one channel (specified by the neurocore) of each kernel. Each kernel will then apply the
        respective weights to a different channel of the current layer.

        @param recurrent A boolean parameter that indicates whether the convolution operation is being
        performed for a recurrent spike or not. If it is True, then the recurrent kernels will
        be used instead of the regular kernels.

        @return The updated neuron states array after performing the convolution operation.

        NOTE: multiply incoming current with (1 - leak)?
        """
        inCurrentLeak = (1-LEAK_RATE)
        kernels = self.recKernels if recurrent else self.kernels
        channels = len(self.neuronStatesConv)
        for c in range(channels):
            self.neuronStatesConv[c,:,:,0] += np.flip(np.flip(kernels[c], axis=0), axis=1)#*inCurrentLeak

        return self.neuronStatesConv

    def checkTreshold(self, neurons : ArrayLike) -> Tuple[ArrayLike, List[Event]]:
        """
        This function checks if the neuron states exceed a threshold potential, resets them if they do,
        and adds a spike event to a queue.

        @param neurons A list of all neurons for the channel of this neurocore.
        @return A tuple containing the updated neuron states and a list of spike events triggered by the
        incoming spike.

        NOTE: Technically doesn't need to be part of the class, but will be performed by the Neurocore in
        Hardware.
        TODO: reset negative states?
        """
        events = []
        for x in range(len(neurons)):
            for y in range(len(neurons[0])):
                if neurons[x, y, 0] > U_THRESH:
                    neurons[x, y, 0] = U_RESET
                    t = neurons[x, y, 1].item()
                    events.append(Event(x, y, t, self.channel))

        return (neurons, events)
