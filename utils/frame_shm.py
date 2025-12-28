from multiprocessing import shared_memory

import numpy as np


class FrameBuffer:
    def __init__(self, name, shape, dtype=np.uint8, create=False):
        self.name = name
        self.shape = tuple(shape)
        self.dtype = dtype
        self.size = int(np.prod(shape) * np.dtype(dtype).itemsize)

        if create:
            try:
                shm = shared_memory.SharedMemory(name=name)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
            self.shm = shared_memory.SharedMemory(
                name=name, create=True, size=self.size
            )
        else:
            self.shm = shared_memory.SharedMemory(name=name)

        self.frame = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def write(self, frame):
        if frame.shape != self.shape:
            raise ValueError(f"Frame shape mismatch: {frame.shape} != {self.shape}")
        if frame.dtype != self.dtype:
            raise ValueError(f"Frame dtype mismatch: {frame.dtype} != {self.dtype}")
        self.frame[:] = frame[:]

    def read(self):
        return self.frame

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()
