import numpy as np
from utils import Spike, Event
from typing import List, Tuple

LEAK_RATE = 0.17
U_RESET = 0
U_THRESH = 1

# class LIFNeuron:
#     def __init__(self, u_init, t_last) -> None:
#         # membrane potential
#         self.u = u_init
#         # timestamp of last leak
#         self.t_last = t_last

#     def applyLeak(self, t_now):
#         t_leak = t_now - self.last_last
#         leak = LEAK_RATE / t_leak
#         self.u = self.u * leak
#         self.last_last = t_now

#     def checkThreshold(self):
#         if (self.u > U_THRESH):
#             self.u = U_RESET
#             return 1
#         else:
#             return 0


def applyLeak(neuron : ArrayLike, t_now) -> ArrayLike:
        """
        This function applies a leak to a neuron based on the time elapsed since the last
        application of the leak.

        @param u The neuron that is being modified by the leak rate as an array containaing
        the membrane potential and the timestamp of the last leak in this order.
        @param t_now The timestamp of the current spike

        @return The modified neuron array.

        TODO: Timestamp update might need to be done at threshold check.
        """
        t_leak = t_now - neuron[1]
        if t_leak*neuron[0] != 0:
            leak = LEAK_RATE / t_leak
            neuron = [neuron[0]*leak, t_now]

        return neuron


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
        for c in range(channels):
            # load neuron states neighbouring spike coordinates
            # start pos-1 stop pos+2 and increment by 1 to account for padding
            n = neurons[c, s.x_pos:s.x_pos+3, s.y_pos:s.y_pos+3]
            self.neuronStatesLeak[c] = n
        self.spikeLeak = s

    def leakNeurons(self):
        """
        This function applies a leak to the neuron states and forwards the spike object to the next
        pipline step performing the convolution.
        """
        channels = len(self.neuronStatesLeak)
        # self.neuronStates.apply_(applyLeak()) Doesn't work because it's applied on both u and t
        for c in range(channels):
            for x in range(self.kernelSize):
                for y in range(self.kernelSize):
                    self.neuronStatesLeak[c, x, y] = applyLeak(self.neuronStatesLeak[c,x,y], self.spikeLeak.timestamp)
        # forward spike and neurons to convolution step
        self.spikeConv = self.spikeLeak
        self.neuronStatesConv = self.neuronStatesLeak

                    u = self.neuronStatesLeak[c,x,y,0].item()
                    t_last = self.neuronStatesLeak[c,x,y,1].item()
                    self.neuronStatesLeak[c, x, y] = applyLeak(u, t_last, self.spikeLeak.timestamp)
        # forward spike and neurons to convolution step
        self.spikeConv = self.spikeLeak
        self.neuronStatesConv = self.neuronStatesLeak

    def applyConv(self):
        """
        This function performs the convolution operations for neurons neighbouring the current spike
        and one channel (specified by the neurocore) of each kernel. Each kernel will then apply the
        respective weights to a different channel of the current layer.

        NOTE: multiply incoming current with (1 - leak)?
        """
        channels = len(self.neuronStatesConv)
        for c in range(channels):
            for x in range(self.kernelSize):
                for y in range(self.kernelSize):
                    self.neuronStatesConv[c, x, y, 0] += self.kernels[c, self.kernelSize-x-1, self.kernelSize-y-1]

        return self.neuronStatesConv

    def checkTreshold(self, neurons) -> Tuple[List, List[Event]]:
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
                    t = neurons[x, y, 1]
                    events.append(Event(x, y, t, self.channel))

        return (neurons, events)
