from torch import nn
import torch

import source.utils.pytorch_tf_ops as pytf


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        return self.loss_fn(logits, target)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss_fn = nn.MSELoss

    def forward(self, logits, target):
        return self.loss_fn(logits, target)


class DetectorLoss(nn.Module):
    '''
    Calculates the Error using the CrossEntropyLoss

    Typically the grid_size is 8 so that can do that mapping in the comments

    '''

    def __init__(self, grid_size):
        super(DetectorLoss, self).__init__()

        self.grid_size = grid_size
        self.s2d = pytf.SpaceToDepth(self.grid_size)

    def forward(self, logits, keypoint_map, valid_mask=None):
        '''

        :param logits: Keypoint output from network is format B,C=65,H/8,W/8
        :param keypoint_map: Ground truth keypoint map is of format 1,H,W
        :param valid_mask:
        :return:
        '''
        # Explanation:
        # Model outputs to size C=65,H/8,W/8 . The 65 channels represent the 8x8 grid in the full scale version, +1
        # for the no keypoint bin

        # Modify keypoint map to correct size
        labels = keypoint_map
        labels = labels[:, None, :, :]
        # Convert from 1xHxW to 64xH/8xW/8
        labels = self.s2d.forward(labels)

        new_shape = labels.shape
        new_shape = torch.Size((new_shape[0], 1, new_shape[2], new_shape[3]))

        # Here we add an extra channel for the no_keypoint_bin i.e channel 65 with all 1(ones)
        # And ensure all the keypoint locations have value 2 not 1
        labels = torch.cat((2 * labels, torch.ones(new_shape, device=labels.device)), dim=1)
        # labels is now size B,C=65,H,W

        # we now take the argmax of the channels dimension. If there was a keypoint at a channel location then it has the
        # value 2 so its index is the max. If instead though as in most cases there is no keypoint then it will return
        # the 65 channel so index 64
        labels = torch.argmax(labels, dim=1)

        # Mask the pixels if bordering artifacts appear
        # valid_mask = torch.ones_like(keypoint_map) if valid_mask is None else valid_mask
        # valid_mask = valid_mask[:, None, :, :].float()
        # valid_mask = self.s2d.forward(valid_mask)
        # valid_mask = torch.prod(valid_mask, dim=1)

        # valid_mask = torch.ones_like(logits) if valid_mask is None else valid_mask
        # valid_mask = valid_mask.float()
        # valid_mask = torch.prod(valid_mask, dim=1)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels.long())
        return loss


class DescriptorLoss(nn.Module):
    def __init__(self, grid_size):
        super(DescriptorLoss, self).__init__()

        self.grid_size = grid_size


def descriptor_loss(descriptors, warped_descriptors, homographies,
                    valid_mask=None, **config):
    # Compute the position of the center pixel of every cell in the image
    (batch_size, Hc, Wc) = tf.unstack(tf.to_int32(tf.shape(descriptors)[:3]))
    coord_cells = tf.stack(tf.meshgrid(
        tf.range(Hc), tf.range(Wc), indexing='ij'), axis=-1)
    coord_cells = coord_cells * config['grid_size'] + config['grid_size'] // 2  # (Hc, Wc, 2)
    # coord_cells is now a grid containing the coordinates of the Hc x Wc
    # center pixels of the 8x8 cells of the image

    # Compute the position of the warped center pixels
    warped_coord_cells = warp_points(tf.reshape(coord_cells, [-1, 2]), homographies)
    # warped_coord_cells is now a list of the warped coordinates of all the center
    # pixels of the 8x8 cells of the image, shape (N, Hc x Wc, 2)

    # Compute the pairwise distances and filter the ones less than a threshold
    # The distance is just the pairwise norm of the difference of the two grids
    # Using shape broadcasting, cell_distances has shape (N, Hc, Wc, Hc, Wc)
    coord_cells = tf.to_float(tf.reshape(coord_cells, [1, 1, 1, Hc, Wc, 2]))
    warped_coord_cells = tf.reshape(warped_coord_cells,
                                    [batch_size, Hc, Wc, 1, 1, 2])
    cell_distances = tf.norm(coord_cells - warped_coord_cells, axis=-1)
    s = tf.to_float(tf.less_equal(cell_distances, config['grid_size'] - 0.5))
    # s[id_batch, h, w, h', w'] == 1 if the point of coordinates (h, w) warped by the
    # homography is at a distance from (h', w') less than config['grid_size']
    # and 0 otherwise

    # Compute the pairwise dot product between descriptors: d^t * d'
    descriptors = tf.reshape(descriptors, [batch_size, Hc, Wc, 1, 1, -1])
    warped_descriptors = tf.reshape(warped_descriptors,
                                    [batch_size, 1, 1, Hc, Wc, -1])
    dot_product_desc = tf.reduce_sum(descriptors * warped_descriptors, -1)
    # dot_product_desc[id_batch, h, w, h', w'] is the dot product between the
    # descriptor at position (h, w) in the original descriptors map and the
    # descriptor at position (h', w') in the warped image

    # Compute the loss
    positive_dist = tf.maximum(0., config['positive_margin'] - dot_product_desc)
    negative_dist = tf.maximum(0., dot_product_desc - config['negative_margin'])
    loss = config['lambda_d'] * s * positive_dist + (1 - s) * negative_dist

    # Mask the pixels if bordering artifacts appear
    valid_mask = tf.ones([batch_size,
                          Hc * config['grid_size'],
                          Wc * config['grid_size']], tf.float32) \
        if valid_mask is None else valid_mask
    valid_mask = tf.to_float(valid_mask[..., tf.newaxis])  # for GPU
    valid_mask = tf.space_to_depth(valid_mask, config['grid_size'])
    valid_mask = tf.reduce_prod(valid_mask, axis=3)  # AND along the channel dim
    valid_mask = tf.reshape(valid_mask, [batch_size, 1, 1, Hc, Wc])

    normalization = tf.reduce_sum(valid_mask) * tf.to_float(Hc * Wc)
    # Summaries for debugging
    # tf.summary.scalar('nb_positive', tf.reduce_sum(valid_mask * s) / normalization)
    # tf.summary.scalar('nb_negative', tf.reduce_sum(valid_mask * (1 - s)) / normalization)
    tf.summary.scalar('positive_dist', tf.reduce_sum(valid_mask * config['lambda_d'] *
                                                     s * positive_dist) / normalization)
    tf.summary.scalar('negative_dist', tf.reduce_sum(valid_mask * (1 - s) *
                                                     negative_dist) / normalization)
    loss = tf.reduce_sum(valid_mask * loss) / normalization
    return loss
