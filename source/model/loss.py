from torch import nn
import torch

import utils.pytorch_tf_ops as pytf
from data_loader.utils.homographies import warp_points_torch



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
        valid_mask = torch.ones_like(keypoint_map) if valid_mask is None else valid_mask
        valid_mask = valid_mask[:, None, :, :].float()
        valid_mask = self.s2d.forward(valid_mask)
        valid_mask = torch.prod(valid_mask, dim=1)

        # valid_mask = torch.ones_like(logits) if valid_mask is None else valid_mask
        # valid_mask = valid_mask.float()
        # valid_mask = torch.prod(valid_mask, dim=1)

        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(logits, labels.long())
        #Pytorch cross entropy weight argument works on the classes not the instances. So we just do the mask
        #multiplication at the end
        valid_loss=loss*valid_mask
        return valid_loss.mean()


class DescriptorLoss(nn.Module):
    def __init__(self, grid_size):
        super(DescriptorLoss, self).__init__()

        self.grid_size = grid_size
        self.s2d = pytf.SpaceToDepth(self.grid_size)




    def forward(self,descriptors, warped_descriptors, homographies,
                        valid_mask=None, **config):
        # Compute the position of the center pixel of every cell in the image
        (batch_size, Hc, Wc) = torch.unbind((descriptors.shape[:3]))
        coord_cells = torch.stack(torch.meshgrid(
            Hc, Wc, indexing='ij'), dim=-1)
        coord_cells = coord_cells * config['grid_size'] + config['grid_size'] // 2  # (Hc, Wc, 2)
        # coord_cells is now a grid containing the coordinates of the Hc x Wc
        # center pixels of the 8x8 cells of the image

        # Compute the position of the warped center pixels
        warped_coord_cells = warp_points_torch(torch.reshape(coord_cells, [-1, 2]), homographies)
        # warped_coord_cells is now a list of the warped coordinates of all the center
        # pixels of the 8x8 cells of the image, shape (N, Hc x Wc, 2)

        # Compute the pairwise distances and filter the ones less than a threshold
        # The distance is just the pairwise norm of the difference of the two grids
        # Using shape broadcasting, cell_distances has shape (N, Hc, Wc, Hc, Wc)
        coord_cells = (torch.reshape(coord_cells, [1, 1, 1, Hc, Wc, 2])).float()
        warped_coord_cells = torch.reshape(warped_coord_cells,
                                        [batch_size, Hc, Wc, 1, 1, 2])
        cell_distances = torch.norm(coord_cells - warped_coord_cells, dim=-1)
        s = cell_distances<= config['grid_size'] - 0.5
        # s[id_batch, h, w, h', w'] == 1 if the point of coordinates (h, w) warped by the
        # homography is at a distance from (h', w') less than config['grid_size']
        # and 0 otherwise

        # Compute the pairwise dot product between descriptors: d^t * d'
        descriptors = torch.reshape(descriptors, [batch_size, Hc, Wc, 1, 1, -1])
        warped_descriptors = torch.reshape(warped_descriptors,
                                        [batch_size, 1, 1, Hc, Wc, -1])
        dot_product_desc = torch.sum(descriptors * warped_descriptors, -1)
        # dot_product_desc[id_batch, h, w, h', w'] is the dot product between the
        # descriptor at position (h, w) in the original descriptors map and the
        # descriptor at position (h', w') in the warped image

        # Compute the loss
        positive_dist = torch.max(0., config['positive_margin'] - dot_product_desc)
        negative_dist = torch.max(0., dot_product_desc - config['negative_margin'])
        loss = config['lambda_d'] * s * positive_dist + (1 - s) * negative_dist

        # Mask the pixels if bordering artifacts appear
        valid_mask = torch.ones([batch_size,
                              Hc * config['grid_size'],
                              Wc * config['grid_size']]).float() \
            if valid_mask is None else valid_mask
        valid_mask = (valid_mask[..., None]).float()  # for GPU
        valid_mask = self.s2d(valid_mask)
        valid_mask = torch.prod(valid_mask, dim=3)  # AND along the channel dim
        valid_mask = torch.reshape(valid_mask, [batch_size, 1, 1, Hc, Wc])

        normalization = torch.sum(valid_mask) * (Hc * Wc).float()
        # Summaries for debugging
        # tf.summary.scalar('nb_positive', tf.reduce_sum(valid_mask * s) / normalization)
        # tf.summary.scalar('nb_negative', tf.reduce_sum(valid_mask * (1 - s)) / normalization)
        # tf.summary.scalar('positive_dist', tf.reduce_sum(valid_mask * config['lambda_d'] *
        #                                                  s * positive_dist) / normalization)
        # tf.summary.scalar('negative_dist', tf.reduce_sum(valid_mask * (1 - s) *
        #                                                  negative_dist) / normalization)
        loss = torch.sum(valid_mask * loss) / normalization
        return loss

def log_sum_exp(self, x):
    b, _ = torch.max(x, 1)
    # b.size() = [N, ], unsqueeze() required
    y = b + torch.log(torch.exp(x - b.unsqueeze(dim=1).expand_as(x)).sum(1))
    # y.size() = [N, ], no need to squeeze()
    return y

def class_select(logits, target):
    # in numpy, this would be logits[:, target].
    batch_size, num_classes = logits.size()
    if target.is_cuda:
        device = target.data.get_device()
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .cuda(device)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    else:
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    return logits.masked_select(one_hot_mask)


def cross_entropy_with_weights(logits, target, weights=None):
    assert logits.dim() == 2
    assert not target.requires_grad
    target = target.squeeze(1) if target.dim() == 2 else target
    assert target.dim() == 1
    loss = log_sum_exp(logits) - class_select(logits, target)
    if weights is not None:
        # loss.size() = [N]. Assert weights has the same shape
        assert list(loss.size()) == list(weights.size())
        # Weight the loss
        loss = loss * weights
    return loss


class CrossEntropyLoss(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self, aggregate='mean'):
        super(CrossEntropyLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate

    def forward(self, input, target, weights=None):
        if self.aggregate == 'sum':
            return cross_entropy_with_weights(input, target, weights).sum()
        elif self.aggregate == 'mean':
            return cross_entropy_with_weights(input, target, weights).mean()
        elif self.aggregate is None:
            return cross_entropy_with_weights(input, target, weights)