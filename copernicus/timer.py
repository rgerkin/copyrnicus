import time
import datetime
import threading

class Timer(object):
    def __init__(self, interval=1):
        self.interval = interval
        self._pause = False
        self.thread = threading.Thread(target=self.run, args=())
        self.thread.daemon = True
        
    def start(self):
        self.thread.start()
        
    def run(self):
        while self.interval:
            if not self._pause:
                print(self.interval)
                print(datetime.datetime.now().__str__() + ' : Start task in the background')
            time.sleep(self.interval)
            
    def pause(self):
        self._pause = True
    
    def resume(self):
        self._pause = False
        
    def stop(self):
        self.pause()
        del self