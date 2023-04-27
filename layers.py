from utils import EventQueue
from neurocore import Neurocore

class ConvLayer:
    layer = None
    recurrent = False
    neurons = None

    def __init__(self, inChannels, numKernels, kernelSize) -> None:
        #self.neurons = neurons
        # generate neurocores
        self.neurocores = [Neurocore(c, numKernels, kernelSize) for c in range(inChannels)]
        self.outQueue = EventQueue(self.layer)

    def assignLayer(self, layer, layerKernels, recurrence, neurons):
        self.layer = layer
        self.neurons = neurons
        self.recurrent = recurrence
        for nc in self.neurocores:
            nc.assignLayer(self.layer, layerKernels)

    def forward(self, inQueue : EventQueue) -> EventQueue:
        while inQueue.qsize() > 0:
            ev = inQueue._get()
            c = ev.channel
            s = ev.toSpike()
            self.neurocores[c].loadNeurons(s, self.neurons)
            self.neurocores[c].leakNeurons()
            self.neurocores[c].applyKernel()
            newEvents = self.neurocores[c].checkTreshold()
            for item in newEvents:
                self.outQueue.put(item)
                # TODO: Recurrence

        return self.outQueue
