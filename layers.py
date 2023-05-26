from typing import List, Tuple
from numpy.typing import ArrayLike
from utils import SpikeQueue
from neurocore import Neurocore


# The ConvLayer class defines a convolutional layer with neurocores that can apply kernels to input
# spikes and generate output spikes.
class ConvLayer:
    recurrent = False
    neurons = None
    outQueue = SpikeQueue()
    recQueue = SpikeQueue()
    inQueue = SpikeQueue()

    def __init__(self, inChannels, numKernels, kernelSize, dtype) -> None:
        """
        This function initializes a list of neurocores for each input channel with a specified number of
        kernels and kernel size.

        @param inChannels The number of input channels to the neural network layer.
        @param numKernels The number of kernels to be used in the convolutional layer.
        @param kernelSize The kernelSize parameter is the size of the convolutional kernel/filter that will
        be used in the convolutional layer.
        """
        # generate neurocores
        self.neurocores = [Neurocore(c, numKernels, kernelSize, dtype) for c in range(inChannels)]

    def assignLayer(self, inQueue : SpikeQueue, layerKernels, neurons, recQueue = SpikeQueue(), recKernels = None):
        """
        This function assigns a new layer to a set of neurocores with specified kernels, and neurons.

        @param inQueue A List containing Spike tuples representing the input queue for the layer.
        @param layerKernels A Numpy array containing the convolutional kernels that represent the
        computation to be performed by each neurocore in the layer. Each neurocore will be assigned
        one channel of each kernel from this list.
        @param neurons A set of neuron states of the new layer
        @param recQueue A List containing Spike tuples representing recurrent spikes of an earlier iteration
        @param recKernels A numpy array containing the ercurrent Kernels, if the layer should be recurrent or
        None otherwise.
        """
        self.inQueue = inQueue
        self.recQueue = recQueue
        self.outQueue = SpikeQueue()
        self.neurons = neurons
        self.recurrent = recKernels is not None
        for nc in self.neurocores:
            nc.assignLayer(layerKernels, recKernels)

    def updateNeurons(self, x, y, updatedNeurons):
        """
        This function writes back the neurons updated by the neurocore to the current layer
        based on the location of the spike that caused the update.

        @param x The x-coordinate of the location where a spike occurred.
        @param y The y-coordinate of the location where a spike occurred.
        @param updatedNeurons a numpy array containing the updated values for a subset of neurons in the
        neural network layer.
        """
        # NOTE: needs to be adjusted for kernels with a size other then c*3*3
        # determines if the spike occured at an edge of the channel
        l = 1 if x > 0 else 0
        r = 1 if x < len(self.neurons[0])-1 else 0
        u = 1 if y > 0 else 0
        d = 1 if y < len(self.neurons[0,0])-1 else 0

        self.neurons[:, x-l:x+r+1, y-u:y+d+1] = updatedNeurons[:, 1-l:2+r, 1-u:2+d]

    def generateSpikes(self, neurons, channel) -> ArrayLike:
        """
        This function generates spikes for a given set of neurons, and returns the updated neuron states.
        The threshold check is performed by the neurocore, which also performed the convolution before.

        @param neurons A numpy array of all neurons, which were updated by the convolution
        @param channel The channel where the spike occured that let to the convolution opeattion.
        Determines which neurocore will perform the threshold check.

        @return The updated neurons as a numpy array.
        """
        (updatedNeurons, newEvents, recEvents) = self.neurocores[channel].checkThreshold(neurons, self.recurrent)
        self.outQueue.extend(newEvents)
        self.recQueue.extend(recEvents)

        return updatedNeurons

    def forward(self, neuronInLog = None, neuronOutLog = None) -> Tuple[List, SpikeQueue, SpikeQueue]:
        """
        This function processes events from an input queue, updates neurons, and generates new events
        for an output queue.

        @param neuronInLog A numpy array used to log all weighted input spikes seperated by channel
        and time bins for the observed neuron.
        @param neuronOutLog A numpy array used to log all output spikes seperated by time bins for the
        observed neuron.

        @return a tuple containing a list of neurons states, an event queue and the timestamp of the last update
        """

        while len(self.inQueue) > 0:
            if (len(self.recQueue) > 0):
                recSpike = (self.inQueue[0].t > self.recQueue[0].t)
            else:
                recSpike = False

            if recSpike:
                s = self.recQueue.pop(0)
            else:
                s = self.inQueue.pop(0)
            c = s.c

            self.neurocores[c].loadNeurons(s, self.neurons)
            self.neurocores[c].leakNeurons()
            updatedNeurons = self.neurocores[c].applyConv(neuronInLog, neuronOutLog, recSpike)
            updatedNeurons = self.generateSpikes(updatedNeurons, c)
            self.updateNeurons(s.x, s.y, updatedNeurons)

        return self.neurons, self.outQueue, self.recQueue
