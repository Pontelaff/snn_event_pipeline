import numpy as np
from utils import Spike, SpikeQueue
from typing import Tuple
from numpy.typing import ArrayLike

LOG_NEURON = (1, 18, 1, 1) #Layer, Channel, x, y
LOG_BINSIZE = 100
LEAK_RATE = 0.90
U_RESET = 0
U_THRESH = 0.4
REC_DELAY = 100
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

def applyLeak(u, t_last, t_now) -> np.float16:
    """
    This function applies a leak to a neuron based on the time elapsed since the last
    application of the leak.

    @param u The membrane potential of the neuron that is being modified by the leak rate.
    @param t_last The timestamp of the last update for this neuron.
    @param t_now The timestamp of the incoming spike.

    @return The leaked membrane potential.

    NOTE: This might be optimised further
    """
    t_leak = t_now - t_last
    # leak neuron if timestamps are different and potential is not zero
    if t_leak*u != 0:
        leak = LEAK_RATE / t_leak
        u = u*leak

    return u


class Neurocore:
    # member attributes
    kernels = None      # 32*3*3 numpy array containing one channel each of 32 Kernels
    recKernels = None   # recurrent Kernels
    spikeLeak = None    # spike for neuron Leak step
    spikeConv = None    # spike for convolution step

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
        self.neuronStatesLeak = np.zeros([numKernels,kernelSize,kernelSize], dtype=dtype) # numpy array containing neighbours of spiking neuron
        self.neuronStatesConv = self.neuronStatesLeak.copy()

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
        @param neurons A 4-dimensional numpy array representing the neurons in a layer. The first
        dimension represents the channel, the second and third dimensions represent the x and y
        positions of the neurons in the layer, the forth dimension contains the neuron states.
        """

        # pad each channel with zeros (don't pad neuron states)
        neurons = np.pad(neurons, ((0,0),(1,1),(1,1)), 'constant')
        # for each channel of current layer
        # load neuron states neighbouring spike coordinates
        # start pos-1 stop pos+2 and increment by 1 to account for padding
        self.neuronStatesLeak = neurons[:, s.x:s.x+3, s.y:s.y+3]
        self.spikeLeak = s

    def leakNeurons(self):
        """
        This function applies a leak to the neuron states and forwards the spike object to the next
        pipline step performing the convolution.
        """
        leakFunc = np.vectorize(applyLeak)
        u = self.neuronStatesLeak['u']
        t = self.neuronStatesLeak['t']
        self.neuronStatesLeak['u'] = leakFunc(u, t, self.spikeLeak.t)
        # forward spike and neurons to convolution step
        self.spikeConv = self.spikeLeak
        self.neuronStatesConv = self.neuronStatesLeak.copy()

    def applyConv(self, neuronInLog = None, neuronOutLog = None, recurrent = False) -> ArrayLike:
        """
        This function performs the convolution operations for neurons neighbouring the current spike
        and one channel (specified by the neurocore) of each kernel. Each kernel will then apply the
        respective weights to a different channel of the current layer.

        @param neuronInLog A numpy array used to log all weighted input spikes seperated by channel
        and time bins for the observed neuron.
        @param neuronOutLog A numpy array used to log all output spikes seperated by time bins for the
        observed neuron.
        @param recurrent A boolean parameter that indicates whether the convolution operation is being
        performed for a recurrent spike or not. If it is True, then the recurrent kernels will
        be used instead of the regular kernels.

        @return The updated neuron states array after performing the convolution operation.

        NOTE: multiply incoming current with (1 - leak)?
        """
        #inCurrentLeak = (1-LEAK_RATE)
        kernels = self.recKernels if recurrent else self.kernels
        #updateIndices = np.where((self.neuronStatesConv['u'] != U_RESET) | (self.neuronStatesConv['t'] < self.spikeConv.t - REFRACTORY_PERIOD))
        #self.neuronStatesConv['u'][updateIndices] += kernels[updateIndices]#*inCurrentLeak
        #self.neuronStatesConv['t'][updateIndices] = self.spikeConv.t

        # updateKernel = np.where(((self.neuronStatesConv['u'] != U_RESET) | (self.neuronStatesConv['t'] < self.spikeConv.t - REFRACTORY_PERIOD)), kernels, 0)
        # updatedTime = np.where((self.neuronStatesConv['u'] != U_RESET) | (self.neuronStatesConv['t'] < self.spikeConv.t - REFRACTORY_PERIOD), self.spikeConv.t, self.neuronStatesConv['t'])
        # self.neuronStatesConv['u'] += updateKernel#*inCurrentLeak
        # self.neuronStatesConv['t'] = updatedTime

        timeMask = self.neuronStatesConv['t'] < (self.spikeConv.t.item() - REFRACTORY_PERIOD)
        resetMask = self.neuronStatesConv['u'] != U_RESET
        updateMask = np.logical_or(timeMask, resetMask)
        potentialUpdates = kernels * updateMask
        timeUpdates = self.spikeConv.t * updateMask + self.neuronStatesConv['t']* np.logical_not(updateMask)
        self.neuronStatesConv['u'] += potentialUpdates
        self.neuronStatesConv['t'] = timeUpdates


        # log neuron activities
        if neuronInLog is not None:
            ln = LOG_NEURON
            x_offset = self.spikeConv.x - ln[2]
            y_offset = self.spikeConv.y - ln[3]
            if areNeighbours(x_offset, y_offset, self.kernelSize):
                bin = self.spikeConv.t//LOG_BINSIZE
                neuronInLog[bin, self.spikeConv.c + int(recurrent) * 32] += kernels[ln[1], x_offset + 1, y_offset+1]
                if (self.neuronStatesConv[ln[1], x_offset + 1, y_offset+1]['u'] >= U_THRESH):
                    neuronOutLog[bin] += 1

        return self.neuronStatesConv

    def checkThreshold(self, neurons : ArrayLike, recurrent = False) -> Tuple[ArrayLike, SpikeQueue, SpikeQueue]:
        """
        This function checks if the neuron states exceed a threshold potential, resets them if they do,
        and adds a spike event to a queue.

        @param neurons A list of all neurons for the channel of this neurocore.
        @param recurrent Determines if recurrent spikes will be generated or not
        @return A tuple containing the updated neuron states, a list of spike events and recurrent spike events
        triggered by the incoming spike.

        NOTE: Technically doesn't need to be part of the class, but will be performed by the Neurocore in
        Hardware.
        NOTE: Returning empty queue for non-recurrent layers isn't optimal on hardware.
        TODO: reset negative states?
        """
        # Get indices of all neurons that exceed U_THRESH
        exceed_indices = np.where(neurons['u'] > U_THRESH)
        # Reset potential of all exceeded neurons
        neurons['u'][exceed_indices] = U_RESET
        # Extract the timestamps of exceeded neurons and create corresponding events
        events = [Spike(x, y, c, neurons[c, x, y]['t']) for c, x, y in zip(*exceed_indices)]
        if recurrent and len(events) > 0:
            recEvents = [Spike(x, y, c, neurons[c, x, y]['t'] + REC_DELAY) for c, x, y in zip(*exceed_indices)]
        else:
            recEvents = SpikeQueue()

        return neurons, events, recEvents
