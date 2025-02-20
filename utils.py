from collections import namedtuple

Spike = namedtuple('Spike', ['x', 'y', 'c', 't'])

class SpikeQueue(list):
    def __init__(self, *args):
        list.__init__(self, *args)

    def pop(self, i = 0) -> Spike:
        return super().pop(0)


def cropLogs(logA, logB):
    length = min(len(logA), len(logB))
    logA = logA[:length]
    logB = logB[:length]

    return logA, logB