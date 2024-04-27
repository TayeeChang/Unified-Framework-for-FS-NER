
import numpy as np
import tensorflow as tf


def nearest_neighbor(support_repr, query_repr, support_labels, mask, metric='euclidean'):
    ndim = support_repr.shape[-1]
    support_repr = support_repr.reshape(-1, ndim)
    support_labels = support_labels.reshape(1, -1)
    if metric == 'cosine':
        logits = _cosine_metric(query_repr, support_repr)
    elif metric == 'euclidean':
        logits = _euclidean_metric(query_repr, support_repr)
    else:
        raise AttributeError
    mask = mask.reshape(1, -1)
    logits = logits * mask - 1e12 * (1 - mask)

    logits = logits.argmax(-1)
    preds = np.take_along_axis(support_labels, logits, axis=-1)
    return preds


def _nearest_neighbor(query_repr, support_repr, support_labels, metric='euclidean'):
    bsz, seq = query_repr.shape[0], query_repr.shape[1]
    if metric == 'euclidean':
        scores = _euclidean_metric_efficient(query_repr.reshape((-1, support_repr.shape[-1])), support_repr, normalize=True)
        query_labels = support_labels[scores.argmax(-1)]
    else:
        raise Exception('undefined metrics for query labels prediction.')

    return query_labels.reshape((bsz, seq))


def _cosine_metric(a, b, normalize=True):
    if normalize:
        a = a / np.linalg.norm(a, axis=-1, keepdims=True)
        b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    dist = np.einsum('bmd, nd -> bmn', a, b)
    return dist


def _euclidean_metric(x, y, normalize=False):
    assert isinstance(x, np.ndarray) and x.ndim == 2
    assert isinstance(y, np.ndarray) and y.ndim == 2
    assert x.shape[1] == y.shape[1]

    if normalize:
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        y = y / np.linalg.norm(y, axis=-1, keepdims=True)

    distance = ((x[:, None] - y[None, :]) ** 2).sum(-1)
    return -1 * distance


def _euclidean_metric_efficient(x, y, squared=True, normalize=True):
    """Compute pairwise (squared) Euclidean distances.
    """
    if x.ndim == 3:
        bsz, seq, ndim = x.shape
        x = x.reshape(-1, ndim)
    assert isinstance(x, np.ndarray) and x.ndim == 2
    assert isinstance(y, np.ndarray) and y.ndim == 2
    assert x.shape[1] == y.shape[1]

    if normalize:
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        y = y / np.linalg.norm(y, axis=-1, keepdims=True)

    x_square = np.sum(x * x, axis=1, keepdims=True)
    if x is y:
        y_square = x_square.T
    else:
        y_square = np.sum(y * y, axis=1, keepdims=True).T
    distances = np.dot(x, y.T)
    # use inplace operation to accelerate
    distances *= -2
    distances += x_square
    distances += y_square
    # result maybe less than 0 due to floating point rounding errors.
    np.maximum(distances, 0, distances)
    if x is y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0
    if not squared:
        np.sqrt(distances, distances)
    if x.ndim == 3:
        distances = distances.reshape(bsz, seq, -1)
    return -1 * distances


def euclidean_metric_efficient(x, y, squared=True, normalize=True):
    """Compute pairwise (squared) Euclidean distances.
    """
    assert x.shape[1] == y.shape[1]

    if normalize:
        x = x / tf.linalg.norm(x, axis=-1, keepdims=True)
        y = y / tf.linalg.norm(y, axis=-1, keepdims=True)

    x_square = tf.reduce_sum(x * x, axis=1, keepdims=True)
    if x is y:
        y_square = tf.transpose(x_square)
    else:
        y_square = tf.transpose(tf.reduce_sum(y * y, axis=1, keepdims=True))

    distances = tf.matmul(x, tf.transpose(y))
    # use inplace operation to accelerate
    distances *= -2
    distances += x_square
    distances += y_square
    # result maybe less than 0 due to floating point rounding errors.
    distances = tf.maximum(distances, 0)
    if x is y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0
    if not squared:
        tf.sqrt(distances, distances)
    # distances = distances.reshape(bsz, seq, -1)
    return -1 * distances

def euclidean_metric(x, y, normalize=True):
    if normalize:
        x = x / tf.linalg.norm(x, axis=-1, keepdims=True)
        y = y / tf.linalg.norm(y, axis=-1, keepdims=True)

    distance = tf.reduce_sum((x - y) ** 2, -1)
    return -distance


def euclidean_metric_3d(x, y, normalize=True):
    if normalize:
        x = x / tf.linalg.norm(x, axis=-1, keepdims=True)
        y = y / tf.linalg.norm(y, axis=-1, keepdims=True)

    distance = tf.reduce_sum((x[:, :, None] - y[:, None, :]) ** 2, -1)
    return -distance



def loss_kl(mu_i, sigma_i, mu_j, sigma_j, embed_dimension):
    sigma_ratio = sigma_j / sigma_i
    trace_fac = tf.reduce_sum(sigma_ratio, 1)
    log_det = tf.reduce_sum(tf.math.log(sigma_ratio + 1e-14), axis=1)
    mu_diff_sq = tf.reduce_sum((mu_i - mu_j) ** 2 / sigma_i, axis=1)
    ij_kl = 0.5 * (trace_fac + mu_diff_sq - embed_dimension - log_det)
    sigma_ratio = sigma_i / sigma_j
    trace_fac = tf.reduce_sum(sigma_ratio, 1)
    log_det = tf.reduce_sum(tf.math.log(sigma_ratio + 1e-14), axis=1)
    mu_diff_sq = tf.reduce_sum((mu_j - mu_i) ** 2 / sigma_j, axis=1)
    ji_kl = 0.5 * (trace_fac + mu_diff_sq - embed_dimension - log_det)
    kl_d = 0.5 * (ij_kl + ji_kl)
    return -kl_d


