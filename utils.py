from collections import namedtuple

Spike = namedtuple('Spike', ['x_pos', 'y_pos', 'channel', 'timestamp'])

class SpikeQueue(list):
    def pop(self, i = 0) -> Spike:
        return super().pop()