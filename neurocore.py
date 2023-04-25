from utils import Spike, Event, EventQueue
import numpy as np

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
        """
        This is the initialization function for a neural network layer, setting various attributes such
        as the channel, neuron states, kernels, and spike leak.

        @param channel The channel parameter is a variable that represents the number of channels in the
        input data. In this case, it is being used to initialize the object's channel attribute.
        """
        self.channel = channel
        self.layer = None
        self.neuronStates = np.zeros([CONV_CHANNELS,KERNEL_SIZE,KERNEL_SIZE,2], dtype=np.float16) # numpy array containing neighbours of spiking neuron
        self.kernels = None # 32*3*3 numpy array containing one channel of 32 Kernels
        self.spikeLeak = None
        self.spikeKernel = None

    def assignLayer(self, newLayer, kernels):
        """
        This function assigns a new layer and loads kernels for that layer based on the channel
        specified for the Neurocore.

        @param newLayer The new layer that the Neurocore will be assigned to.
        @param kernels All kernals of the neural network as a numpy array
                (Layers*Kernals*Channels*KSize*KSize)
        """
        self.layer = newLayer
        # load kernels
        self.kernels = kernels[newLayer, :, self.channel]

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
            self.neuronStates[c] = n
        self.spikeLeak = s

    def applyLeak(self, u, t_last, t_now) -> np.array:
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

    def leakNeurons(self):
        """
        This function applies a leak to the neuron states and forwards the spike object to the next
        pipline step performing the convolution.
        """
        channels = len(self.neuronStates)
        # self.neuronStates.apply_(applyLeak()) Doesn't work because it's applied on both u and t
        for c in range(channels):
            for x in range(KERNEL_SIZE):
                for y in range(KERNEL_SIZE):
                    u = self.neuronStates[c,x,y,0].item()
                    t_last = self.neuronStates[c,x,y,1].item()
                    self.neuronStates[c, x, y] = self.applyLeak(u, t_last, self.spikeLeak.timestamp)
        # forward spike to convolution step
        self.spikeKernel = self.spikeLeak

    def applyKernel(self):
        """
        This function performs the convolution operations for neurons neighbouring the current spike
        and one channel (specified by the neurocore) of each kernel. Each kernel will then apply the
        respective weights to a different channel of the current layer.
        """
        channels = len(self.neuronStates)
        for c in range(channels):
            for x in range(KERNEL_SIZE):
                for y in range(KERNEL_SIZE):
                    self.neuronStates[c, x, y, 0] += self.kernels[c, KERNEL_SIZE-x-1, KERNEL_SIZE-y-1]

    def checkTreshold(self) -> EventQueue:
        """
        This function checks if the neuron states exceed a threshold potential and resets them if they do, while
        also adding a spike event to a queue.

        @return an EventQueue object containing all spikes triggered by the incoming spike.
        """
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
                        if min(x_pos, y_pos) >= 0 and max(x_pos, y_pos <=31):
                            queue.put(Event(x_pos, y_pos, t, c))

        return queue



allKernels = np.random.rand(7, CONV_CHANNELS, CONV_CHANNELS, KERNEL_SIZE, KERNEL_SIZE).astype(np.float16)
allNeurons = np.ones([7, CONV_CHANNELS, SEG_WIDTH, SEG_HEIGHT, 2], dtype=np.float16)
spike = Spike(0,0,12)

# pad each channel with zeros (don't pad neuron states)
layer1Neurons = np.pad(allNeurons[1], ((0,0),(1,1),(1,1),(0,0)), 'constant')

# recognizable test values
allKernels[1,0,0,0] = [1,2,3]
allKernels[1,1,0,0] = [44,55,66]
allKernels[1,2,0,1] = [0,55,0]
layer1Neurons[0,11,23:26] = [[0,1],[2,3],[55,10]]

nc = Neurocore(0)
nc.assignLayer(1, allKernels)
nc.loadNeurons(spike, layer1Neurons)

nc.leakNeurons()
nc.applyKernel()
outQ = nc.checkTreshold()

pass