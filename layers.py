from typing import List, Tuple
from utils import EventQueue, Spike
from neurocore import Neurocore
# The ConvLayer class defines a convolutional layer with neurocores that can apply kernels to input
# spikes and generate output spikes.

class ConvLayer:
    #layer = None
    recurrent = False
    neurons = None
    outQueue = None
    inQueue = None

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

    def assignLayer(self, inQueue : EventQueue, layerKernels, neurons, recurrence):
        """
        This function assigns a new layer to a set of neurocores with specified kernels, and neurons.

        @param inQueue An EventQueue object that represents the input queue for the layer.
        @param layerKernels This parameter is a list of kernel objects that represent the computation to
        be performed by each neurocore in the layer. Each neurocore will be assigned one channel of each
        kernel from this list.
        @param neurons A set of neuron states of the new layer
        @param recurrence A boolean value indicating whether the layer has recurrent connections or not.
        """

        self.inQueue = inQueue
        self.outQueue = EventQueue()
        self.neurons = neurons
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
        """
        for nc in range(len(self.neurocores)):
            (self.neurons[nc], newEvents) = self.neurocores[nc].checkTreshold(self.neurons[nc])
            for item in newEvents:
                self.outQueue.put(item)
                # TODO: Recurrence

    def forward(self, recQueue : EventQueue) -> Tuple[List, EventQueue]:
        """
        This function processes events from an input queue, updates neurons, and generates new events
        for an output queue.

        @param recQueue The recQueue parameter is an EventQueue object that represents the queue of
        events that need to be processed recursively.

        @return a tuple containing a list of neurons states and an event queue.
        """
        t = 0

        for _ in range(self.inQueue.qsize()):
            ev = self.inQueue._get()
            c = ev.channel
            s = ev.toSpike()
            if t < s.timestamp:
                # next timestamps, generate Spikes for last timestamp
                self.generateSpikes()
                t = s.timestamp

            self.inQueue.task_done()
            self.neurocores[c].loadNeurons(s, self.neurons)
            self.neurocores[c].leakNeurons()
            updatedNeurons = self.neurocores[c].applyConv()
            self.updateNeurons(s.x_pos, s.y_pos, updatedNeurons)

        return (self.neurons, self.outQueue)
