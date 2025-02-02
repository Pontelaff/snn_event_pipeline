import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
from utils import SpikeQueue, Spike
from pipeline import Pipeline
from config import EVENT_TIMESLICE

INPUT_LEAKS = True
LEAK_RATE = 0.017
REC_DELAY = EVENT_TIMESLICE
U_RESET = 0
U_THRESH = 1.0

# The Neurocore class represents a convolutional layer that can apply kernels to input
# spikes and generate output spikes.
class Neurocore:
    recurrent = False
    neurons = None
    thresholds = None
    leaks = None
    timestamp = None
    outQueue = SpikeQueue()
    recQueue = SpikeQueue()
    inQueue = SpikeQueue()

    def __init__(self, inChannels, numKernels, kernelSize, dtype) -> None:
        """
        This function initializes a list of pipelines for each input channel with a specified number of
        kernels and kernel size.

        @param inChannels The number of input channels to the neurocore.
        @param numKernels The number of kernels to be used in neurocore.
        @param kernelSize The kernelSize parameter is the size of the convolutional kernel/filter that will
        be used in the neurocore.
        """
        # Generate pipelines
        self.pipelines = [Pipeline(c, numKernels, kernelSize, dtype) for c in range(inChannels)]

    def assignLayer(self, inQueue: SpikeQueue, layerKernels, neurons, recQueue = SpikeQueue(), recKernels = None, threshold = None, leak = None):
        """
        This function assigns new layer parameters to a set of pipelines with specified kernels and neurons.

        @param inQueue A List containing Spike tuples representing the input queue for the neurocore.
        @param layerKernels A Numpy array containing the convolutional kernels that represent the
        computation to be performed by each pipeline of the neurocore. Each pipeline will be assigned
        one channel of each kernel from this list.
        @param neurons A set of neuron states of the new layer
        @param recQueue A List containing Spike tuples representing recurrent spikes of an earlier iteration
        @param recKernels A numpy array containing the rercurrent Kernels, if the layer should be recurrent or
        None otherwise.
        @param threshold A 1-dimensional array containing per channel thresholds.
        @param threshold A 1-dimensional array containing per channel leak rates.
        """
        numOutChannels = len(layerKernels)
        self.inQueue = inQueue
        self.recQueue = recQueue
        self.outQueue = SpikeQueue()
        self.neurons = neurons
        self.recurrent = recKernels is not None
        self.timestamp = (inQueue[0].t//EVENT_TIMESLICE)*EVENT_TIMESLICE

        # Use default threshold and leak rate if none are given
        if threshold is not None:
            self.thresholds = threshold
        else:
            self.thresholds = np.ones([numOutChannels, 1, 1]) * U_THRESH
        if leak is not None:
            self.leaks = leak
        else:
            self.leaks = np.ones([numOutChannels, 1, 1]) * LEAK_RATE

        # Multiply input kernels with (1 - leak), if INPUT_LEAKS is set
        # The original implementation multiplied all incoming currents with (1 - leak)
        # Modifying the kernels once achieves the same result
        if INPUT_LEAKS:
            layerKernels = layerKernels * (1 - self.leaks.reshape(numOutChannels,1,1,1))
            if self.recurrent:
                recKernels = recKernels * (1 - self.leaks.reshape(numOutChannels,1,1,1))

        # load kernels into pipelines
        for nc in self.pipelines:
            nc.assignLayer(layerKernels, recKernels)

    def leakNeurons(self):
        """
        This function applies a channelwise leak to the neuron states.
        """
        self.neurons = self.neurons*self.leaks

        return self.neurons

    def checkThreshold(self, neuronOutLog = None, neuronStateLog = None, loggedNeuron  = None) -> Tuple[ArrayLike, SpikeQueue, SpikeQueue]:
        """
        This function checks if the neuron states exceed a threshold potential, resets them if they do,
        and adds a spike event to a queue.

        @param neuronOutLog A numpy array used to log the output spikes of one neuron in the layer at each
        channel and time step.
        @param neuronStateLog A numpy array used to log the membrane potential of one neuron in the layer at each
        channel and time step.
        @param loggedNeuron  A tuple that contains the channel, row, and column indices of a specific neuron in
        the network. It is used to log the state and output of that neuron.
        """

        # Log neuron output spikes
        if loggedNeuron  is not None:
            logNeuronStates = self.neurons[:,loggedNeuron [1],loggedNeuron [2]]
            thresh = np.reshape(self.thresholds, np.shape(logNeuronStates))
            neuronStateLog[:] = logNeuronStates
            neuronOutLog[:] = np.where(logNeuronStates > thresh, 1, 0)

        # Get indices of all neurons that exceed the threshold
        exceed_indices = np.where(self.neurons >= self.thresholds)

        # Reset potential of all exceeded neurons
        self.neurons[exceed_indices] = U_RESET
        #self.neurons[exceed_indices] -= self.thresholds # soft reset

        # Generate events for spiking neurons
        ffEvents = [Spike(x, y, c, self.timestamp) for c, x, y in zip(*exceed_indices)]
        if self.recurrent and (len(ffEvents) > 0):
            recEvents = [Spike(x, y, c, self.timestamp + REC_DELAY) for c, x, y in zip(*exceed_indices)]
        else:
            recEvents = SpikeQueue()

        # Add events to queue
        self.outQueue.extend(ffEvents)
        self.recQueue.extend(recEvents)

    def updateNeurons(self, x, y, updatedNeurons):
        """
        This function writes back the neurons updated by the pipeline to the current layer
        based on the location of the spike that caused the update.

        @param x The x-coordinate of the location where a spike occurred.
        @param y The y-coordinate of the location where a spike occurred.
        @param updatedNeurons a numpy array containing the updated values for a subset of neurons in the
        neurocore.
        """
        # NOTE: Assumes 3x3 kernels. Adjust for other kernel sizes.
        # Determines if the spike occured at an edge of the channel
        l = 1 if x > 0 else 0
        r = 1 if x < len(self.neurons[0])-1 else 0
        u = 1 if y > 0 else 0
        d = 1 if y < len(self.neurons[0,0])-1 else 0

        self.neurons[:, x-l:x+r+1, y-u:y+d+1] = updatedNeurons[:, 1-l:2+r, 1-u:2+d]

    def forward(self, neuronInLog = None, neuronOutLog = None, neuronStateLog = None, loggedNeuron = None) -> Tuple[List, SpikeQueue, SpikeQueue, SpikeQueue]:
        """
        This function processes events from an input queue, updates neurons, and generates new events
        for an output queue.

        @param neuronInLog A numpy array used to log all weighted input spikes seperated by channel
        and time bins for the observed neuron.
        @param neuronOutLog A numpy array used to log all output spikes seperated by time bins for the
        observed neuron.
        @param loggedNeuron The `loggedNeuron` parameter is an optional argument that specifies a single
        neuron whose activity should be logged. If this parameter is not provided, no neuron activity
        will be logged.
        @param threshold The threshold to use in the pipeline. If 'None' the default threshold constant
        will be used.

        @return a tuple containing a list of neurons states, and event queues
        """

        while len(self.inQueue) > 0:
            if (len(self.recQueue) > 0):
                spikeIsRec = (self.inQueue[0].t > self.recQueue[0].t)
            else:
                spikeIsRec = False

            if spikeIsRec:
                s = self.recQueue.pop(0)
            else:
                s = self.inQueue.pop(0)

            if (s.t >= self.timestamp + EVENT_TIMESLICE):
                # Next time slice reached, generate spikes and leak neurons
                # Only reached for input layer to seperate event input into time windows
                self.checkThreshold(neuronOutLog, neuronStateLog, loggedNeuron)
                self.leakNeurons()
                self.timestamp = (s.t//EVENT_TIMESLICE)*EVENT_TIMESLICE
                break

            updatedNeurons = self.pipelines[s.c].forward(s, self.neurons, spikeIsRec, neuronInLog, loggedNeuron)
            self.updateNeurons(s.x, s.y, updatedNeurons)

        # End of queue reached, generate spikes and leak neurons
        if len(self.inQueue) == 0:
            self.checkThreshold(neuronOutLog, neuronStateLog, loggedNeuron)
            self.leakNeurons()

        return self.neurons, self.inQueue, self.outQueue, self.recQueue
