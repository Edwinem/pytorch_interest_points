from torch import nn
import torch


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


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size,input_height,output_width,output_depth) for t_t in spl]
        output = torch.stack(stacks,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size,output_height,output_width,output_depth)
        output = output.permute(0, 3, 1, 2)
        return output

class DetectorLoss(nn.Module):
    def __init__(self,grid_size):
        super(DetectorLoss,self).__init__()

        self.grid_size=grid_size
        self.s2d=SpaceToDepth(8)

    def forward(self,logits,keypoint_map, valid_mask=None):
        def detector_loss(logits,keypoint_map, valid_mask=None, **config):
            # Convert the boolean labels to indices including the "no interest point" dustbin
            labels = tf.to_float(keypoint_map[..., tf.newaxis])  # for GPU
            labels = tf.space_to_depth(labels, config['grid_size'])
            shape = tf.concat([tf.shape(labels)[:3], [1]], axis=0)
            labels = tf.argmax(tf.concat([2 * labels, tf.ones(shape)], 3), axis=3)

            # Mask the pixels if bordering artifacts appear
            valid_mask = tf.ones_like(keypoint_map) if valid_mask is None else valid_mask
            valid_mask = tf.to_float(valid_mask[..., tf.newaxis])  # for GPU
            valid_mask = tf.space_to_depth(valid_mask, config['grid_size'])
            valid_mask = tf.reduce_prod(valid_mask, axis=3)  # AND along the channel dim

            loss = tf.losses.sparse_softmax_cross_entropy(
                labels=labels, logits=logits, weights=valid_mask)

        labels=keypoint_map
        labels=labels[:,None,:,:]
        labels=self.s2d.forward(labels)
        uno=torch.tensor([1],device=labels.device)
        new_shape=labels.shape
        new_shape=torch.Size((new_shape[0],1,new_shape[2],new_shape[3]))
        labels=torch.cat((2*labels,torch.ones(new_shape,device=labels.device)),dim=1)
        labels=torch.argmax(labels,dim=1)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits,labels.long())
        return loss




class DescriptorLoss(nn.Module):
    def __init__(self,grid_size):
        super(DescriptorLoss,self).__init__()

        self.grid_size=grid_size

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
                          Wc * config['grid_size']], tf.float32)\
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
