import torch
import torch.nn as nn
import torch.nn.functional as F

from caloGraphNN import *



def gather_features_from_conv_head(conv_head, vertices_y, vertices_x, vertices_y2, vertices_x2, scale_y, scale_x):
    """
    Gather features from a 2D image.

    :param conv_head: The 2D conv head with shape [batch, height, width, channels]
    :param vertices_y: The y position of each of the vertex with shape [batch, max_vertices]
    :param vertices_x: The x position of each of the vertex with shape [batch, max_vertices]
    :param vertices_height: The height of each of the vertex with shape [batch, max_vertices]
    :param vertices_width: The width of each of the feature with shape [batch, max_vertices]
    :param scale_y: A scalar to show y_scale
    :param scale_x: A scalar to show x_scale
    :return: The gathered features with shape [batch, max_vertices, channels]
    """
    conv_head = conv_head.permute(0,2,3,1)

    vertices_y = vertices_y * scale_y
    vertices_x = vertices_x * scale_x
    vertices_y2 = vertices_y2 * scale_y
    vertices_x2 = vertices_x2 * scale_x

    batch_size, max_vertices = vertices_y.shape

    batch_range =  torch.arange(0, batch_size)[:, None]
    y_indices = ((vertices_y + vertices_y2 ) /2).long()
    x_indices = ((vertices_x + vertices_x2 ) /2).long()

    return conv_head[batch_range, y_indices, x_indices, :]


class EdgeConv(nn.Module):
    def __init__(self, num_neighbors=10, input_size=64, mlp_layers=[128,64,64], aggregation_function=torch.max, initializer=None, bias_initializer=None):
        super(EdgeConv, self).__init__()
        self.num_neighbors = num_neighbors

        self.aggregation_function = aggregation_function

        mlp_layers.insert(0, input_size*2)  # -> mlp_layers=[128,128,64,64]

        self.bn = nn.BatchNorm1d(input_size, momentum=0.01)

        layers = []
        for i in range(1, len(mlp_layers)):
            layers.append(nn.Linear(mlp_layers[i - 1], mlp_layers[i]))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

        self.global_exchange_layer = nn.Sequential(
                nn.Linear(mlp_layers[-1] * 2, mlp_layers[-1]),
                nn.ReLU()
            )

    def forward(self, input_space):
        shape = input_space.shape # -> [b,900,64]
        input_space = self.bn(input_space.view(-1, input_space.shape[-1])).view(*input_space.shape)

        neighbour_space = gather_neighbours(input_space, self.num_neighbors) # -> [b,900,64]
        expanded_input_space = input_space[:, :, None, :].repeat(1, 1, self.num_neighbors, 1) # -> [b,900,10,64]
        edge = torch.cat((expanded_input_space, expanded_input_space - neighbour_space), dim=-1) # -> [b,900,10,128]

        edge = self.mlp(edge)
        vertex_out, _ = self.aggregation_function(edge, dim=2)  # [b,900,10,128] -> [b,900,128]

        global_summed, _ = torch.max(vertex_out, dim=1, keepdims=True) # [b,900,128] -> [b,1,128]
        global_concat = torch.cat((vertex_out, global_summed.repeat(1, input_space.shape[1], 1)), dim=-1) # -> [b,900,128]

        global_exchanged = self.global_exchange_layer(global_concat) # [b,900,128] -> [b,900,64]

        return global_exchanged