import queue as q

class Spike:
    def __init__(self, x, y, t):
        self.x_pos = x
        self.y_pos = y
        self.timestamp = t

class Event(Spike):
    def __init__(self, x, y, t, c):
        super().__init__(x, y, t)
        self.channel = c

    def toSpike(self) -> Spike:
        return Spike(self.x_pos, self.y_pos, self.timestamp)

class EventQueue(q.Queue): # NOTE: PriorityQueue for rec?
    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize)
        #self.layer = layer

    def _get(self) -> Event:
        return super()._get()


# class Kernel:
#     def __init__(self, depth=32, size=3):
#         self.weights = torch.zeros([depth,size,size], dtype=torch.float16)

#     def getChannel(self, c):
#         return self.weights[c]
