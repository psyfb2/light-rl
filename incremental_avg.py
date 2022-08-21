class IncrementalAvg:
    def __init__(self):
        self.reset()
    
    def update(self, val: float):
        self.n += 1
        self.avg = ((self.n - 1) * self.avg + val) / self.n
    
    def reset(self):
        self.avg = 0
        self.n = 0