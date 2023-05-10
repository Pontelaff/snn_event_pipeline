from typing import List, Tuple
from utils import SpikeQueue
from neurocore import Neurocore


# The ConvLayer class defines a convolutional layer with neurocores that can apply kernels to input
# spikes and generate output spikes.
class ConvLayer:
    recurrent = False
    neurons = None
    outQueue = None
    inQueue = None
    t = 0

    def __init__(self, inChannels, numKernels, kernelSize) -> None:
        """
        This function initializes a list of neurocores for each input channel with a specified number of
        kernels and kernel size.

        @param inChannels The number of input channels to the neural network layer.
        @param numKernels The number of kernels to be used in the convolutional layer.
        @param kernelSize The kernelSize parameter is the size of the convolutional kernel/filter that will
        be used in the convolutional layer.
        """
        # generate neurocores
        self.neurocores = [Neurocore(c, numKernels, kernelSize) for c in range(inChannels)]

    def assignLayer(self, inQueue : SpikeQueue, layerKernels, neurons, t_last, recurrence):
        """
        This function assigns a new layer to a set of neurocores with specified kernels, and neurons.

        @param inQueue A List containing Spike tuples representing the input queue for the layer.
        @param layerKernels This parameter is a list of kernel objects that represent the computation to
        be performed by each neurocore in the layer. Each neurocore will be assigned one channel of each
        kernel from this list.
        @param neurons A set of neuron states of the new layer
        @param recurrence A boolean value indicating whether the layer has recurrent connections or not.
        """
        self.inQueue = inQueue
        self.outQueue = []
        self.neurons = neurons
        self.t = t_last
        self.recurrent = recurrence
        for nc in self.neurocores:
            nc.assignLayer(layerKernels)

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

    def generateSpikes(self):
        """
        This function applies the threshold check for all neurons of each channel and adds resulting
        events to the output queue.

        NOTE: There are two other alternatives, which check thresholds and generate spikes with each
        incoming spike. However, measures must be taken to prohibit spiking during the same and
        potentially following timestamps (depending on refractory time).
        This can be done by:
        1. Marking spiked neurons with a flag or counter. This might improve performance but increases
        the amount of data stored and transmitted.
        2. Skipping the convolution step, if the timestamp is up to date (or within refractory period)
        and the membrane potential is at reset value, indicating a spike just occured. For that to work,
        leaks (and therefore timestamp updates) can not be applied to neurons already at reset potential,
        which also means that timestamps then need be updated at threshold check.
        """
        for nc in range(len(self.neurocores)):
            (self.neurons[nc], newEvents) = self.neurocores[nc].checkThreshold(self.neurons[nc])
            self.outQueue.extend(newEvents)
            # TODO: Recurrence

    def forward(self, recQueue : SpikeQueue = None) -> Tuple[List, SpikeQueue]:
        """
        This function processes events from an input queue, updates neurons, and generates new events
        for an output queue.

        @param recQueue The recQueue parameter is an SpikeQueue list object that represents the queue of
        events that need to be processed recursively.

        @return a tuple containing a list of neurons states, an event queue and the timestamp of the last update
        """
        #TODO t musst be timestamp of last incoming spike for the layer

        for _ in range(len(self.inQueue)):
            s = self.inQueue.pop(0)
            c = s.channel
            if self.t < s.timestamp:
                # next timestamps, generate Spikes for last timestamp
                self.generateSpikes()
                self.t = s.timestamp

            self.neurocores[c].loadNeurons(s, self.neurons)
            self.neurocores[c].leakNeurons()
            updatedNeurons = self.neurocores[c].applyConv(self.t)
            self.updateNeurons(s.x_pos, s.y_pos, updatedNeurons)

        return self.neurons, self.outQueue, self.t
