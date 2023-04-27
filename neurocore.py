import numpy as np
from utils import Spike, Event, EventQueue
from typing import List

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


def applyLeak(u, t_last, t_now) -> np.array:
        """
        This function applies a leak to a neuron based on the time elapsed since the last
        application of the leak.

        @param u The neuron potential that is being modified by the leak rate.
        @param t_last The time stamp of the last applied leak for this neuron.
        @param t_now The timestamp of the current spike

        @return a numpy array with two elements: the updated neuron potential
        and the timestamp of the current spike `t_now`.
        """
        t_leak = t_now - t_last
        leak = LEAK_RATE / t_leak
        u = u * leak
        return np.array([u, t_now], dtype=np.float16)


class Neurocore:

    # member attributes
    layer = None        # the currently active layer
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

    def assignLayer(self, newLayer, kernels):
        """
        This function assigns a new layer and loads kernels for that layer based on the channel
        specified for the Neurocore.

        @param newLayer The new layer that the Neurocore will be assigned to.
        @param kernels All kernals of the neural network layer as a numpy array
                [Kernals, Channels, KSize, KSize]
        """
        self.layer = newLayer-1
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
                    u = self.neuronStatesLeak[c,x,y,0].item()
                    t_last = self.neuronStatesLeak[c,x,y,1].item()
                    self.neuronStatesLeak[c, x, y] = applyLeak(u, t_last, self.spikeLeak.timestamp)
        # forward spike and neurons to convolution step
        self.spikeConv = self.spikeLeak
        self.neuronStatesConv = self.neuronStatesLeak

    def applyKernel(self):
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

    def checkTreshold(self) -> List[Event]:
        """
        This function checks if the neuron states exceed a threshold potential and resets them if they do, while
        also adding a spike event to a queue.

        @return an EventQueue object containing all spikes triggered by the incoming spike.

        TODO: reset negative states?
        """
        events = []#EventQueue(self.layer)
        channels = len(self.neuronStatesConv)
        # self.neuronStatesConv.apply_(checkTresh()) Doesn't work because it's applied on both u and t
        for c in range(channels):
            for x in range(self.kernelSize):
                for y in range(self.kernelSize):
                    if self.neuronStatesConv[c, x, y, 0] > U_THRESH:
                        self.neuronStatesConv[c, x, y, 0] = U_RESET
                        x_pos = self.spikeConv.x_pos + x -1
                        y_pos = self.spikeConv.y_pos + y -1
                        t = self.spikeConv.timestamp
                        if min(x_pos, y_pos) >= 0 and max(x_pos, y_pos <=31):
                            events.append(Event(x_pos, y_pos, t, c))
                            #queue.put(Event(x_pos, y_pos, t, c))

        return events
