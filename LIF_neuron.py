import time
import matplotlib.pyplot as plt
import numpy as np

U_REST = 0
U_THRESH = 5
TAU = 100
T_REFRACTORY = 100
NEGATIVE_U = True

class LIF_neuron:
    def __init__(self, u_init=0) -> None:
        # membrane potential
        self.u = u_init
        # timestamp of last update
        self.last_update = time.time()
        # timestamp of last spike
        self.last_spike = 0

    def update_LIF_neuron(self, weight=1):
        spike = 0
        u_next = self.u
        # timestamp of incoming spike
        t_now = time.time()

        # only update after refractory period
        if (t_now - self.last_spike) * 1000 > T_REFRACTORY:
            if self.u != U_REST:
                t_passed = (t_now - self.last_update) * 1000
                # decay depending on time since last update
                decay_rate = 1 - np.exp(-t_passed/TAU)
                # membrane potential decay
                self.u = self.u - (self.u - U_REST) * decay_rate

            # increase membrane potential by weight of incoming spike
            u_next = self.u + weight
            if u_next >= U_THRESH:
                # send spike and reset to resting potential
                spike = 1
                self.last_spike = t_now
                self.u = U_REST
            else:
                # update membrane potential
                if u_next < 0 and not NEGATIVE_U:
                    self.u = 0
                    u_next = 0
                else:
                    self.u = u_next

        self.last_update = t_now

        return (u_next, spike)

# demonstrates the membrane potential decay of a neuron
def LIF_decay_demo(length):
    potential_map = np.zeros(length)
    spike_map = np.zeros(length)
    timestamps = np.zeros(length)

    # initialize neuron just below threshold
    n = LIF_neuron(U_THRESH * 0.99)

    t_start = n.last_update
    potential_map[0] = n.u
    spike_map[0] = 0
    timestamps[0] = 0

    # simulate potential decay
    for i in range(length -1):
        time.sleep(TAU*0.0001)
        (potential_map[i + 1], spike_map[i + 1]) = n.update_LIF_neuron(0)
        timestamps[i + 1] = (n.last_update - t_start) *1000

    plot_LIF(timestamps, potential_map, spike_map, "Neuron membrane potential decay")

# demonstrates the response of a neuron to incoming spikes
def LIF_spike_demo(length):
    potential_map = np.zeros(length)
    spike_map = np.zeros(length)
    timestamps = np.zeros(length)

    # initialize neuron
    n = LIF_neuron()
    t_start = time.time()

    # update the neuron by simulating incoming spikes with randomized weights and timing
    for i in range(length):
        (potential_map[i], spike_map[i]) = n.update_LIF_neuron(np.random.randn()+0.5)
        timestamps[i] = (n.last_update - t_start) *1000
        time.sleep(np.random.random()*0.001)

    plot_LIF(timestamps, potential_map, spike_map, "Neuron membrane potential with random input spikes")


def plot_LIF(t, potential, spikes, title):
    plt.figure()
    plt.subplot(211)
    plt.plot(t, potential)
    #plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential")
    plt.title(title)
    plt.subplot(212)
    #plt.plot(t, spikes)
    plt.vlines(t, 0, spikes)
    plt.xlabel("Time (ms)")
    plt.ylabel("Output spikes")

LIF_decay_demo(50)
LIF_spike_demo(200)

plt.show()
