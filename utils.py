from collections import namedtuple

Spike = namedtuple('Spike', ['x', 'y', 'c', 't'])

class SpikeQueue(list):
    def pop(self, i = 0) -> Spike:
        return super().pop()