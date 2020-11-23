from queue import Queue
from threading import Thread, Lock
import os
import numpy as np
import pickle
import gzip


class InferenceOutputStreamer:
    def __init__(self, output_path, cache_size=50):
        self._cache_size = cache_size
        self._cache = list() # It will store `cache` number of elements
        self._closed = False
        self._initialized = False
        self._output_path = output_path
        self._num_last_file = 0
        self._all_files = []

        if not os.path.exists(output_path):
            os.mkdir(output_path)

    def _worker(self):
        while True:
            output_cache = self._queue.get()
            file_path = os.path.join(self._output_path, 'data_'+ str(self._num_last_file)+ '.bin')
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(output_cache, f)
            self._all_files.append(file_path)
            self._num_last_file += 1
            self._queue.task_done()
            del output_cache

    def start_thread(self):
        self._queue = Queue() # When we are out of `cache` number of elements in cache, push it to queue, so it could be written
        self._t = Thread(target=self._worker)
        self._t.setDaemon(True)
        self._t.start()
        self._initialized=True

    def add(self, sample):
        if self._closed or (not self._initialized):
            RuntimeError("Attempting to use a closed or an unopened streamer")
        indices = np.where(sample['image']!=255)
        sample['image'] = sample['image'][:np.amax(indices[0]), :np.amax(indices[1])]
        self._cache.append(sample)
        n = len(self._cache)
        if n == self._cache_size:
            self._queue.put(self._cache)
            self._cache = list()

    def close(self):
        self._queue.put(self._cache)
        self._cache = list()

        print("X", self._queue.qsize())

        self._queue.join()

        with open(os.path.join(self._output_path, 'inference_output_files.txt'), 'w') as f:
            for file_path in self._all_files:
                f.write("%s\n" % file_path)

        self._closed = True
