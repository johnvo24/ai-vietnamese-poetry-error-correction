import time

class JTimer():
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.times.append(time.time() - self.start_time)
        return self.times[-1]
    
    def sum(self):
        return sum(self.times)
    
    def avg(self):
        return sum(self.times)/len(self.times)