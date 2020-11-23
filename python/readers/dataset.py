import os
import torch
import numpy as np
from readers.GenerateTFRecord import GenerateTFRecord
from torch.utils.data import IterableDataset
from torchvision import transforms, utils
from libs.configuration_manager import ConfigurationManager as conf

class TableDataset(IterableDataset):
    def __init__(self, root_dir, device, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.generator = GenerateTFRecord(os.path.join(root_dir, "images"),
                                        os.path.join(root_dir, "ocr"), 
                                        os.path.join(root_dir, "gt"),
                                        False, False,
                                        device)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = 1
        else:
            per_worker = 1 / float(worker_info.num_workers)
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, 1)
        return iter(self.generator.data_generator(iter_start, iter_end))