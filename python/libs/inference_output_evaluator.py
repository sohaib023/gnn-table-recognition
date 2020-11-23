import csv
from os import path, mkdir
from queue import Queue
from threading import Thread, Lock

import cv2
import gzip
import pickle
import numpy as np

from tqdm import tqdm
import networkx as nx
from libs.Rect import Rect

class Block(Rect):
    def __init__(self, x0=0, y0=0, x1=1, y1=1, w_ids=None, label=0, cells=None):
        super().__init__(x0, y0, x1, y1)
        self.label = -1
        self.w_ids = w_ids
        self.cells = cells

    def copy(self):
        return Block(self.x1, self.y1, self.x2, self.y2, w_ids=self.w_ids.copy(), label=self.label, cells=self.cells)

class InferenceOutputEvaluator:
    def __init__(self, output_path):
        self._output_path = output_path
        self._output_images1 = path.join(output_path, "Adjacency_level")
        self._all_files = []

        self.adj_metrics = {
            'right': {'tp':0, 'fp':0,'tn':0, 'fn':0}, 
            'down':  {'tp':0, 'fp':0,'tn':0, 'fn':0}
        }

        if not path.exists(self._output_path):
            mkdir(self._output_path)
        if not path.exists(self._output_images1):
            mkdir(self._output_images1)

    def evaluate(self):
        if not path.isfile(path.join(self._output_path, 'inference_output_files.txt')):
            raise Exception("'inference_output_files.txt' not found. Please run inference first to resolve this error.")

        with open(path.join(self._output_path, 'inference_output_files.txt'), 'r') as f:
            self._all_files = [line.replace('\n', '') for line in f.readlines()]

        data = None
        self.idx = 0
        for filename in self._all_files:
            print("Reading: ", filename)
            with gzip.open(filename, "rb") as f:
                data = pickle.load(f)

            with tqdm(total=len(data)) as pbar:
                for sample in data:
                    self.process_sample(sample)
                    pbar.update(1)
                    self.idx += 1

        with open(path.join(self._output_path, "evaluation.csv"), "w") as f:
            csv_writer = csv.writer(f, delimiter=',')

            csv_writer.writerow([''] + [key for key in self.adj_metrics])
            csv_writer.writerow(['Adjacency level evaluation'])
            csv_writer.writerow(['True Positives'] + [self.adj_metrics[key]['tp'] for key in self.adj_metrics])
            csv_writer.writerow(['True Negatives'] + [self.adj_metrics[key]['tn'] for key in self.adj_metrics])
            csv_writer.writerow(['False Positives'] + [self.adj_metrics[key]['fp'] for key in self.adj_metrics])
            csv_writer.writerow(['False Negatives'] + [self.adj_metrics[key]['fn'] for key in self.adj_metrics])

            csv_writer.writerow(['Precision'] + 
                                            [str(round(self.adj_metrics[key]['tp'] * 100/ (self.adj_metrics[key]['tp'] + self.adj_metrics[key]['fp'] + 1e-8), 2)) + '%'
                                             for key in self.adj_metrics])
            csv_writer.writerow(['Recall'] + 
                                            [str(round(self.adj_metrics[key]['tp'] * 100/ (self.adj_metrics[key]['tp'] + self.adj_metrics[key]['fn'] + 1e-8), 2)) + '%'
                                             for key in self.adj_metrics])

    def process_sample(self, sample):
        image = cv2.cvtColor(sample["image"], cv2.COLOR_BGR2GRAY)
        sampled_ground_truths = sample["sampled_ground_truths"]
        sampled_predictions = sample["sampled_predictions"]

        h, w, _, _, _, _ = sample["global_features"]
        h, w =  int(h), int(w)

        self.vertex_features = sample["vertex_features"]

        right_mask = sample["masks"][0]
        down_mask = sample["masks"][1]

        rows = sample["rows"]
        columns = sample["columns"]
        n_rows = len(rows) - 1
        n_columns = len(columns) - 1

        sampled_ground_truths[0] = sampled_ground_truths[0][:n_rows, :n_columns-1]
        sampled_ground_truths[1] = sampled_ground_truths[1][:n_rows-1, :n_columns]
        
        sampled_predictions[0] = sampled_predictions[0][:n_rows, :n_columns-1]
        sampled_predictions[1] = sampled_predictions[1][:n_rows-1, :n_columns]

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for row in rows:
            cv2.line(image, (0, row), (image.shape[1], row), (0, 255, 0), 3)
        for col in columns:
            cv2.line(image, (col, 0), (col, image.shape[0]), (0, 255, 0), 3)

        cv2.imwrite(path.join(self._output_images1, str(self.idx) + ".png"), image)

        for adj_name, predictions, gt in zip(
            ["right", "down"], sampled_predictions, sampled_ground_truths
        ):
            img = image.copy()

            for y in range(predictions.shape[0]):
                for x in range(predictions.shape[1]):
                    if adj_name == "right":
                        bbox1_x = int(columns[x] + columns[x + 1]) // 2
                        bbox1_y = int(rows[y] + rows[y + 1])//2

                        bbox2_x = int(columns[x + 1] + columns[x + 2]) // 2
                        bbox2_y = bbox1_y
                    else:
                        bbox1_x = int(columns[x] + columns[x + 1]) // 2
                        bbox1_y = int(rows[y] + rows[y + 1])//2

                        bbox2_x = bbox1_x
                        bbox2_y = int(rows[y + 1] + rows[y + 2])//2

                    if predictions[y, x] == 1:
                        cv2.line(img, (bbox1_x, bbox1_y), (bbox2_x, bbox2_y), (0, 0, 255), 3)
                    if gt[y, x] == 1:
                        cv2.line(img, (bbox1_x-5, bbox1_y-5), (bbox2_x-5, bbox2_y-5), (255, 0, 0), 3)

            true_positives = np.count_nonzero((predictions == 1) & (gt == 1))
            false_positives = np.count_nonzero((predictions == 1) & (gt == 0))
            false_negatives = np.count_nonzero((predictions == 0) & (gt == 1))
            true_negatives = np.count_nonzero((predictions == 0) & (gt == 0))

            self.adj_metrics[adj_name]['tp'] += true_positives
            self.adj_metrics[adj_name]['fp'] += false_positives
            self.adj_metrics[adj_name]['tn'] += true_negatives
            self.adj_metrics[adj_name]['fn'] += false_negatives

            scale = 0.6
            color = (0, 180, 50)
            h = img.shape[0]

            cv2.putText(img, "True Positives: " + str(true_positives), (0, h-20), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2) 
            cv2.putText(img, "True Negatives: " + str(true_negatives), (0, h-40), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2) 
            cv2.putText(img, "False Positives: " + str(false_positives), (0, h-60), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2) 
            cv2.putText(img, "False Negatives: " + str(false_negatives), (0, h-80), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2) 
            
            cv2.putText(img, "Precision: " + str(true_positives * 100/ max(1, true_positives + false_positives)) + "%", (350, h-70), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
            cv2.putText(img, "Recall: " + str(true_positives * 100/ max(1, true_positives + false_negatives)) + "%", (350, h-40), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

            cv2.imwrite(path.join(self._output_images1, str(self.idx) + '-' + adj_name + '.png'), img)