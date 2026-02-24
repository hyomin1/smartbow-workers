import signal
import time


class BaseWorker:
    def __init__(self, name: str):
        self.name = name
        self.running = True

        signal.signal(signal.SIGTERM, self._stop)
        signal.signal(signal.SIGINT, self._stop)

    def _stop(self, *args):
        print(f"[{self.name}] SIGTERM received")
        self.running = False

    def sleep(self, sec=0.01):
        time.sleep(sec)
