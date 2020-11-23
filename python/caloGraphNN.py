import torch
import torch.nn.functional as F

### helpers here


def gauss(x):
    return torch.exp(-1* x*x)

def gauss_of_lin(x):
    return torch.exp(-1*(torch.abs(x)))

def euclidean_squared(A, B):
    """
    Returns euclidean distance between two batches of shape [B,N,F] and [B,M,F] where B is batch size, N is number of
    examples in the batch of first set, M is number of examples in the batch of second set, F is number of spatial
    features.

    Returns:
    A matrix of size [B, N, M] where each element [i,j] denotes euclidean distance between ith entry in first set and
    jth in second set.

    """

    # shape_A = A.get_shape().as_list()
    # shape_B = B.get_shape().as_list()
    
    # assert (A.dtype == tf.float32 or A.dtype == tf.float64) and (B.dtype == tf.float32 or B.dtype == tf.float64)
    # assert len(shape_A) == 3 and len(shape_B) == 3
    # assert shape_A[0] == shape_B[0]# and shape_A[1] == shape_B[1]



    # Finds euclidean distance using property (a-b)^2 = a^2 + b^2 - 2ab
    sub_factor = -2 * torch.matmul(A, B.permute(0, 2, 1))  # -2ab term
    dotA = torch.sum(A * A, dim=2).unsqueeze(2)  # a^2 term
    dotB = torch.sum(B * B, dim=2).unsqueeze(1)  # b^2 term
    return torch.abs(sub_factor + dotA + dotB)


def nearest_neighbor_matrix(spatial_features, k=10):
    """
    Nearest neighbors matrix given spatial features.

    :param spatial_features: Spatial features of shape [B, N, S] where B = batch size, N = max examples in batch,
                             S = spatial features
    :param k: Max neighbors
    :return:
    """

    # shape = spatial_features.shape

    # assert spatial_features.dtype == tf.float32 or spatial_features.dtype == tf.float64
    # assert len(shape) == 3

    D = euclidean_squared(spatial_features, spatial_features)

    D, N = torch.topk(-D, k, dim=-1)
    
    return N, -D


def gather_neighbours(spatial_features, k=10):

    shape_spatial_features = spatial_features.shape
    n_batch = shape_spatial_features[0]
    n_max_entries = shape_spatial_features[1]

    # All of these tensors should be 3-dimensional
    # assert len(shape_spatial_features) == 3

    # # Neighbor matrix should be int as it should be used for indexing
    # assert spatial_features.dtype == tf.float64 or spatial_features.dtype == tf.float32

    neighbor_matrix, distance_matrix = nearest_neighbor_matrix(spatial_features, k)

    batch_indices = torch.arange(0, n_batch)[:, None, None]

    neighbor_space = spatial_features[batch_indices, neighbor_matrix, :]
    return spatial_features[batch_indices, neighbor_matrix, :]