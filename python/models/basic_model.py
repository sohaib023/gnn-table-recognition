import os
import random
from overrides import overrides

import cv2
import numpy as np

from caloGraphNN import *

from ops.ties import *
from models.dgcnn_segment import DgcnnSegment
from libs.configuration_manager import ConfigurationManager as gconfig
from libs.inference_output_streamer import InferenceOutputStreamer
from libs.visual_feedback_generator import VisualFeedbackGenerator

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class BasicModel(torch.nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

        self.max_rows = gconfig.get("max_rows", int)
        self.max_columns = gconfig.get("max_columns", int)

        # self.initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        # self.bias_initializer = tf.constant_initializer(.1)

        self.image_height = gconfig.get_config_param("max_image_height", "int")
        self.image_width = gconfig.get_config_param("max_image_width", "int")

        self.dim_num_vertices = gconfig.get_config_param("dim_num_vertices", "int")
        
        self.conv_segment = self.get_backbone(torchvision.models.resnet18(pretrained=True))
        self.upscale = nn.Sequential(
                nn.Linear(5, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 64),
                nn.LeakyReLU()
            )

        self.downscale = nn.Sequential(
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU()
            )

        self.graph_segment = DgcnnSegment()
        self.graph_segment_after_pooling = DgcnnSegment(input_size=192, num_neighbors=4)
        # self.bns = nn.ModuleList([
        #     nn.BatchNorm1d(128, momentum=0.01)
        #     for i in range(2)
        # ])
        self.bn = nn.BatchNorm1d(192, momentum=0.01)
        
        self.classification_heads2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(384,256),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5),

                nn.Linear(256,128),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5),

                nn.Linear(128,64),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5),

                nn.Linear(64,32),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5),

                nn.Linear(32, 2),
                nn.ReLU()
            )
            for i in range(2)
        ])

    def get_backbone(self, model):
        modules = list(model.named_children())[:-3]  # delete the avg and the fc layers.
        modules.remove(modules[3])
        backbone = nn.Sequential()
        for key, module in modules:
            backbone.add_module(key, module)
        return backbone

    def apply_conv(self, images, vertex_features):
        b, c, image_width, image_height = images.shape
        conv_head = self.conv_segment(images)

        vertices_x = vertex_features[:, :, 0]
        vertices_y = vertex_features[:, :, 1]
        vertices_x2 = vertex_features[:, :,2]
        vertices_y2 = vertex_features[:, :,3]

        b, c, post_height, post_width = conv_head.shape

        scale_y = post_height / image_height
        scale_x = post_width / image_width
        return gather_features_from_conv_head(conv_head, vertices_y, vertices_x,
                                                                 vertices_y2, vertices_x2, scale_y, scale_x)

    def forward(self, images, vertex_features, cell_ids):

        conv_features = self.apply_conv(images, vertex_features)

        vertices_combined_features = torch.cat((self.upscale(vertex_features), self.downscale(conv_features)), dim=-1)

        graph_features = self.graph_segment(vertices_combined_features)
        shape = graph_features.shape
        graph_features = self.bn(graph_features.view(-1, shape[-1])).view(*shape)

        pooled_features = self.grid_pooling(graph_features, cell_ids)
        
        # orig_pooled_features_shape = pooled_features.shape
        # pooled_features = self.graph_segment_after_pooling(pooled_features.view(pooled_features.shape[0], -1, pooled_features.shape[3]))
        # pooled_features = pooled_features.view(*orig_pooled_features_shape)

        right_sampled = torch.cat((pooled_features[:, :, :-1, :], pooled_features[:, :, 1:, :]), dim=-1)
        down_sampled =  torch.cat((pooled_features[:, :-1, :, :], pooled_features[:, 1:, :, :]), dim=-1)

        sampled_features_list = [right_sampled, down_sampled]

        results = []
        for i in range(2):
            pair_sampled_features = sampled_features_list[i]
            shape = pair_sampled_features.shape
            # normalized = self.bns[i](pair_sampled_features.view(-1, shape[-1])).view(*shape)
            out = self.classification_heads2[i](pair_sampled_features)
            results.append(out)

        return tuple(results)

    def grid_pooling(self, graph_features, cell_ids):
        # sampled = torch.zeros(graph_features.shape[0], )
        # for sample_num in range(graph_features.shape[0]):
        #     for i in range(rows.shape[1]):
        #         row_mask = vertex_features[sample_num, 0] < rows[sample_num]
        #         for j in range(columns.shape[1]):

        pooled_features = torch.zeros(graph_features.shape[0], self.max_rows - 1, self.max_columns - 1, graph_features.shape[-1]).to(graph_features.device)

        for i in range(pooled_features.shape[1]):
            for j in range(pooled_features.shape[2]):
                mask = (cell_ids[:, :, 0] == i) & (cell_ids[:, :, 1] == j)

                for batch_no in range(graph_features.shape[0]):
                    # if there is some element inside cell
                    if graph_features[batch_no, mask[batch_no],:].shape[0]:
                        pooled_features[batch_no, i, j, :] = torch.max(graph_features[batch_no, mask[batch_no],:], dim=0)[0]
                    # print(pooled_features.shape)
                # print(mask[0][mask[0]==True].shape, mask.shape)
                # x = torch.max(graph_features * mask[:, :, None] + ~mask[:, :, None] * (-100000), dim=-2)[0]
                # pooled_features[:, i, j, :] = x

        return pooled_features

    # def wrap_up(self):
    #     if self.training:
    #         pass
    #     else:
    #         self.inference_output_streamer.close()

