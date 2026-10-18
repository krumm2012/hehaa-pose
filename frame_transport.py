"""Borrow source slots only while their coordinator retains ownership."""
from contextlib import contextmanager
from multiprocessing import shared_memory
import struct

import numpy as np


class FrameTransportReader:
    def __init__(self):
        self.handles = {}
        self.generations = {}

    @contextmanager
    def read(self, task):
        descriptor = task.get('shared_frame')
        if descriptor is None:
            yield task['frame']
            return
        name = descriptor['name']
        shape = tuple(descriptor['shape'])
        generation = descriptor['generation']
        if (len(shape) != 3 or shape[2] != 3 or min(shape) <= 0
                or generation != task['frame_id'] or descriptor['dtype'] != 'uint8'):
            raise ValueError('Invalid shared source frame descriptor')
        size = int(np.prod(shape))
        if name not in self.handles:
            self.handles[name] = shared_memory.SharedMemory(name=name)
        shm = self.handles[name]
        if shm.size < size + 8:
            raise ValueError('Shared source frame lacks generation header')
        def verify():
            if struct.unpack_from('<q', shm.buf, size)[0] != generation:
                raise ValueError('Shared source slot was overwritten')
        verify()
        if generation <= self.generations.get(name, -1):
            raise ValueError('Replayed shared source generation')
        view = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
        view.flags.writeable = False
        try:
            yield view
            verify()
            self.generations[name] = generation
        finally:
            del view

    def close(self):
        for shm in self.handles.values():
            shm.close()
        self.handles.clear()
