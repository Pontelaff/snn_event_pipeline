from typing import List, Tuple
from utils import EventQueue
from neurocore import Neurocore

class ConvLayer:
    #layer = None
    recurrent = False
    neurons = None
    outQueue = None
    inQueue = None

    def __init__(self, inChannels, numKernels, kernelSize) -> None:
        #self.neurons = neurons
        # generate neurocores
        self.neurocores = [Neurocore(c, numKernels, kernelSize) for c in range(inChannels)]

    def assignLayer(self, inQueue : EventQueue, layerKernels, neurons, recurrence):
        self.inQueue = inQueue
        self.outQueue = EventQueue()
        self.neurons = neurons
        self.recurrent = recurrence
        for nc in self.neurocores:
            nc.assignLayer(layerKernels)

    def forward(self, recQueue : EventQueue) -> Tuple[List, EventQueue]:
        for _ in range(self.inQueue.qsize()):
            ev = self.inQueue._get()
            c = ev.channel
            s = ev.toSpike()
            self.neurocores[c].loadNeurons(s, self.neurons)
            self.neurocores[c].leakNeurons()
            self.neurocores[c].applyKernel()
            newEvents = self.neurocores[c].checkTreshold()
            for item in newEvents:
                self.outQueue.put(item)
                # TODO: Recurrence
            self.inQueue.task_done()

        return (self.neurons, self.outQueue)
