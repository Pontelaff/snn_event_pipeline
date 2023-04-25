from utils import Spike, Event, EventQueue
import torch

CONV_CHANNELS = 32
KERNEL_SIZE = 3
SEG_WIDTH = 32
SEG_HEIGHT = 32
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

class Neurocore:
    def __init__(self, channel) -> None:
        self.channel = channel
        self.layer = None
        self.neuronStates = torch.zeros([32,KERNEL_SIZE,KERNEL_SIZE,2]) # Tensor containing neighbours of spiking neuron
        self.kernels = None # 32*3*3 Tensor containing one channel of 32 Kernels
        self.spikeLeak = None
        self.spikeKernel = None

    def assignLayer(self, newLayer, kernels):
        self.layer = newLayer
        # load kernels
        self.kernels = kernels[newLayer, :, self.channel]

    def loadNeurons(self, s: Spike, neurons):
        channels = len(neurons)
        # for each channel of current layer
        for c in range(channels):
            # load neuron states neighbouring spike coordinates
            n = neurons[c, s.x_pos-1:s.x_pos+2, s.y_pos-1:s.y_pos+2]
            self.neuronStates[c] = n
        self.spikeLeak = s

    def applyLeak(u, t_last, t_now) -> torch.HalfTensor:
        t_leak = t_now - t_last
        leak = LEAK_RATE / t_leak
        u = u * leak
        return torch.HalfTensor([u, t_now])

    def leakNeurons(self):
        channels = len(self.neuronStates)
        # self.neuronStates.apply_(applyLeak()) Doesn't work because it's applied on both u and t
        for c in range(channels):
            for x in range(KERNEL_SIZE):
                for y in range(KERNEL_SIZE):
                    self.neuronStates[c, x, y] = self.applyLeak(self.neuronStates[c,x,y,0], self.neuronStates[c,x,y,1], self.spikeLeak.timestamp)
        # forward spike to kernel module
        self.spikeKernel = self.spikeLeak

    def applyKernel(self):
        channels = len(self.neuronStates)
        for c in range(channels):
            for x in range(KERNEL_SIZE):
                for y in range(KERNEL_SIZE):
                    self.neuronStates[c, x, y, 0] += self.kernels[c, KERNEL_SIZE-x-1, KERNEL_SIZE-y-1]

    def checkTreshold(self):
        queue = EventQueue(self.layer)
        channels = len(self.neuronStates)
        # self.neuronStates.apply_(checkTresh()) Doesn't work because it's applied on both u and t
        for c in range(channels):
            for x in range(KERNEL_SIZE):
                for y in range(KERNEL_SIZE):
                    if self.neuronStates[c, x, y, 0] > U_THRESH:
                        self.neuronStates[c, x, y, 0] = U_RESET
                        x_pos = self.spikeKernel.x_pos + x -1
                        y_pos = self.spikeKernel.y_pos + y -1
                        t = self.spikeKernel.timestamp
                        queue.put(Event(x_pos, y_pos, t, c))

        return queue



allKernels = torch.rand([7, CONV_CHANNELS, CONV_CHANNELS, KERNEL_SIZE, KERNEL_SIZE], dtype=torch.float16)
allNeurons = torch.ones([7, CONV_CHANNELS, SEG_WIDTH, SEG_HEIGHT, 2], dtype=torch.float16)
spike = Spike(12,24,12)

layer1Neurons = allNeurons[1]

# recognizable test values
allKernels[1,0,0,0] = torch.HalfTensor([1,2,3])
layer1Neurons[0,11,23:26] = torch.HalfTensor([[0,1],[2,3],[55,10]])
allKernels[1,1,0,0] = torch.HalfTensor([44,55,66])

nc = Neurocore(0)
nc.assignLayer(1, allKernels)
nc.loadNeurons(spike, layer1Neurons)

nc.leakNeurons()
nc.applyKernel()
outQ = nc.checkTreshold()

pass