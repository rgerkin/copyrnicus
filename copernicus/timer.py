import collections
import time
import datetime
import threading

class Timer(object):
    def __init__(self, action, args=(), kwargs={}, postaction=None, interval=1):
        self.action = action
        self.args = args
        self.kwargs = kwargs
        self.interval = interval
        self.results = collections.deque(maxlen=100)
        self.results.append(0)
        self._pause = False
        self.thread = threading.Thread(target=self._run, args=())
        self.thread.daemon = True
        
    def start(self):
        self.thread.start()
        
    def _run(self):
        while self.interval:
            if not self._pause:
                result = self.action(*self.args, **self.kwargs)
                if result > max(self.results):
                    print("New file found.")
                self.results.append(result)
                
            time.sleep(self.interval)
            
    def pause(self):
        self._pause = True
    
    def resume(self):
        self._pause = False
        
    def stop(self):
        self.pause()
        del self