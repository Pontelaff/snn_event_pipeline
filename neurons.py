import utils

LEAK_CONST = 0.17
U_RESET = 0
U_THRESH = 1

class LIFNeuron:
    def __init__(self, u_init=0) -> None:
        # membrane potential
        self.u = u_init
        # timestamp of last leak
        self.t_last = 0

    def applyLeak(self, t_now):
        t_leak = t_now - self.last_last
        leak = LEAK_CONST / t_leak
        self.u = self.u * leak
        self.last_last = t_now

    def checkThreshold(self):
        if (self.u > U_THRESH):
            self.u = U_RESET
            return 1
        else:
            return 0

class Channel:
    def __init__(self, layer, channel) -> None:
        self.layer = layer
        self.channel = channel


    def loadNeurons(s: utils.Spike):
        pass