import time

class Timer:
    def __init__(self, num_seconds):
        self.num_seconds = num_seconds
        self.i = None
        
    def __len__(self):
        if self.i is None: return 1
        eps = 1e-7
        t = time.time() - self.start
        if t >= self.num_seconds: return self.i
    
        return int((self.i*self.num_seconds)/t)
    
    def __iter__(self):
        self.start = time.time()
        self.i = 0
        t = 0.0
        while t <= self.num_seconds:
            t = time.time() - self.start
            self.i += 1
            yield None
