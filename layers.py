from typing import List, Tuple
from utils import EventQueue, Spike
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

    def updateNeurons(self, x, y, updatedNeurons):
        # NOTE: needs to be adjusted for kernels with a size other then c*3*3
        # determines if the spike occured at an edge of the channel
        l = 1 if x > 0 else 0
        r = 1 if x < len(self.neurons[0])-1 else 0
        u = 1 if y > 0 else 0
        d = 1 if y < len(self.neurons[0,0])-1 else 0

        self.neurons[:, x-l:x+r+1, y-u:y+d+1] = updatedNeurons[:, 1-l:2+r, 1-u:2+d]

    def forward(self, recQueue : EventQueue) -> Tuple[List, EventQueue]:
        for _ in range(self.inQueue.qsize()):
            ev = self.inQueue._get()
            c = ev.channel
            s = ev.toSpike()
            self.neurocores[c].loadNeurons(s, self.neurons)
            self.neurocores[c].leakNeurons()
            self.neurocores[c].applyKernel()
            (updatedNeurons, newEvents) = self.neurocores[c].checkTreshold()
            self.updateNeurons(s.x_pos, s.y_pos, updatedNeurons)
            for item in newEvents:
                self.outQueue.put(item)
                # TODO: Recurrence
            self.inQueue.task_done()

        return (self.neurons, self.outQueue)
