from queue import Queue
from  threading import Thread
from os import path
import gzip
import pickle
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from libs.configuration_manager import ConfigurationManager as gconfig
import  time
import sys


class VisualFeedbackGenerator:
    def __init__(self, output_path):
        self._closed = False
        self._initialized = False
        self._output_path = output_path
        self._queue = None
        self._t = None

    def _worker(self):
        while True:
            iteration_number, data = self._queue.get()
            self.work(iteration_number, data)
            self._queue.task_done()




    def start_thread(self):
        self._queue = Queue()  # When we are out of `cache` number of elements in cache, push it to queue, so it could be written
        self._t = Thread(target=self._worker)
        self._t.setDaemon(True)
        self._t.start()
        self._initialized = True

    def add(self, iteration_number, result):
        if self._closed or (not self._initialized):
            RuntimeError("Attempting to use a closed or an unopened streamer")

        self._queue.put((iteration_number, result))

    def close(self):
        self._queue.join()
        self._closed = True

    def work(self, iteration_number, data):
        image = data['image'] # [max_height, max_width]
        gts = data['sampled_ground_truths'] # 3 x [max_entries, num_samples]
        preds = data['sampled_predictions'] # 3 x [max_entries, num_samples]
        sampled_indices = data['sampled_indices'] # [max_entries, num_samples]
        num_vertices = data['global_features'][gconfig.get_config_param("dim_num_vertices", "int")]
        height, width = data['global_features'][gconfig.get_config_param("dim_height", "int")],\
                        data['global_features'][gconfig.get_config_param("dim_width", "int")]

        max_height, max_width = gconfig.get_config_param("max_image_height", "float"),\
                                gconfig.get_config_param("max_image_width", "float")

        height_inches = 10*max_height/max_width

        vertices = data['vertex_features'] # [max_entries, num_vertex_features]

        height, width = int(height), int(width)
        num_vertices = int(num_vertices)

        image = image.astype(np.uint8)

        image = image[:, :, 0]

        assert len(gts) == 3 and len(preds) == 3

        # image = image[0:height, 0:width]
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        file_path = lambda type, it: os.path.join(self._output_path, ("%05d_%s.pdf") % (it,type))

        output_file_name_cells = file_path('cells', iteration_number)
        output_file_name_rows = file_path('rows', iteration_number)
        output_file_name_cols = file_path('cols', iteration_number)
        output_file_paths = [output_file_name_cells, output_file_name_rows, output_file_name_cols]

        x1s = vertices[:, gconfig.get_config_param("dim_vertex_x_position", "int")]
        y1s = vertices[:, gconfig.get_config_param("dim_vertex_y_position", "int")]
        x2s = vertices[:, gconfig.get_config_param("dim_vertex_x2_position", "int")]
        y2s = vertices[:, gconfig.get_config_param("dim_vertex_y2_position", "int")]

        test_samples = np.random.randint(0, num_vertices, size=10)

        color_1 = (0, 0, 255) # (Blue)
        color_2 = (0, 255, 0) # (Green)
        color_3 = (255, 0, 0) # (Red)
        color_4 = (255, 51, 153) # (Pink)
        color_5 = (160, 80, 0) # (Orange)

        for type in range(3):
            _pred = preds[type]
            _gt = gts[type]
            samples = sampled_indices[type]

            samples_per_vertex = samples.shape[1]

            for sample in range(len(test_samples)):
                sample_index = test_samples[sample]
                image_copy = image.copy()

                for i in range(samples_per_vertex):
                    sample_index_pair = samples[sample_index, i]
                    if _pred[sample_index, i] == 0 and  _gt[sample_index, i] == 0: # Blue
                        # print("here 1")
                        cv2.rectangle(image_copy, (x1s[sample_index_pair], y1s[sample_index_pair]), (x2s[sample_index_pair], y2s[sample_index_pair]), color_1, 3)
                    elif _pred[sample_index, i] == 1 and  _gt[sample_index, i] == 1: # Green
                        # print("here 2")
                        cv2.rectangle(image_copy, (x1s[sample_index_pair], y1s[sample_index_pair]), (x2s[sample_index_pair], y2s[sample_index_pair]), color_2, 3)
                    elif _pred[sample_index, i] == 1 and  _gt[sample_index, i] == 0: # Red
                        # print("here 3")
                        cv2.rectangle(image_copy, (x1s[sample_index_pair], y1s[sample_index_pair]), (x2s[sample_index_pair], y2s[sample_index_pair]), color_3, 3)
                    elif _pred[sample_index, i] == 0 and  _gt[sample_index, i] == 1: # Pink
                        # print("here 4")
                        cv2.rectangle(image_copy, (x1s[sample_index_pair], y1s[sample_index_pair]), (x2s[sample_index_pair], y2s[sample_index_pair]), color_4, 3)
                    else:
                        assert False

                cv2.rectangle(image_copy, (x1s[sample_index], y1s[sample_index]), (x2s[sample_index], y2s[sample_index]), color_5, 10)

                plt.figure(figsize=(10, height_inches), dpi=300)
                plt.imshow(image_copy)

            pdf = matplotlib.backends.backend_pdf.PdfPages(output_file_paths[type])
            for fig in range(1, plt.gcf().number + 1):  ## will open an empty extra figure :(
                pdf.savefig(fig)
            pdf.close()
            plt.close('all')
            print("PDF written")


