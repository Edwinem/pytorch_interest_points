import torch
import numpy as np
import cv2


def truncated_normal(shape,mean,std_dev):
    arr=np.empty(shape,dtype=np.float32)

    with np.nditer(arr,op_flags=['readwrite']) as it:
        for x in it:
            while True:
                val=np.random.normal(mean,std_dev)
                dev=(val-mean)/std_dev
                if np.fabs(dev)<2:
                    x[...]=val
                    break
    return arr


def sample_homography(
        shape, perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
        perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=np.pi/2,
        allow_artifacts=False, translation_overflow=0.):
    """Sample a random valid homography.
    Computes the homography transformation between a random patch in the original image
    and a warped projection with the same image size.
    As in `tf.contrib.image.transform`, it maps the output point (warped patch) to a
    transformed input point (original patch).
    The original patch, which is initialized with a simple half-size centered crop, is
    iteratively projected, scaled, rotated and translated.
    Arguments:
        shape: A rank-2 `Tensor` specifying the height and width of the original image.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography.
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.
    Returns:
        A `Tensor` of shape `[1, 8]` corresponding to the flattened homography transform.
    """

    # Corners of the output image
    pts1 = np.stack([[0., 0.], [0., 1.], [1., 1.], [1., 0.]], axis=0)
    # Corners of the input patch
    margin = (1 - patch_ratio) / 2
    pts2 = margin + np.array([[0, 0], [0, patch_ratio],
                                 [patch_ratio, patch_ratio], [patch_ratio, 0]],
                                dtype=np.float32)

    # Random perspective and affine perturbations
    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        perspective_displacement = truncated_normal([1], 0., perspective_amplitude_y/2)
        h_displacement_left = truncated_normal([1], 0., perspective_amplitude_x/2)
        h_displacement_right = truncated_normal([1], 0., perspective_amplitude_x/2)
        pts2 += np.stack([np.concatenate([h_displacement_left, perspective_displacement], 0),
                          np.concatenate([h_displacement_left, -perspective_displacement], 0),
                          np.concatenate([h_displacement_right, perspective_displacement], 0),
                          np.concatenate([h_displacement_right, -perspective_displacement],
                                    0)])

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        scales = np.concatenate(
                [[1.], truncated_normal([n_scales], 1, scaling_amplitude/2)], 0)
        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = np.expand_dims(pts2 - center, axis=0) * np.expand_dims(
                np.expand_dims(scales, 1), 1) + center
        if allow_artifacts:
            valid = np.arange(n_scales)  # all scales are valid except scale=1
        else:
            a=np.all((scaled >= 0.) & (scaled < 1.),(1, 2))
            valid = np.where(np.all((scaled >= 0.) & (scaled < 1.), (1, 2)))[:][ 0]
        idx = valid[np.random.random_integers(0,high=np.shape(valid)[0])]
        pts2 = scaled[idx]

    # Random translation
    if translation:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += np.expand_dims(np.stack([np.random.uniform( -t_min[0], t_max[0]),
                                         np.random.uniform(-t_min[1], t_max[1])]),
                               axis=0)

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = np.linspace(-max_angle, max_angle, n_angles)
        angles = np.concatenate([[0.], angles], axis=0)  # in case no rotation is valid
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul(
                np.tile(np.expand_dims(pts2 - center, axis=0), [n_angles+1, 1, 1]),
                rot_mat) + center
        if allow_artifacts:
            valid = np.arange(n_angles)  # all angles are valid, except angle=0
        else:
            valid = np.where(np.all((rotated >= 0.) & (rotated < 1.),
                                           axis=(1, 2)))[:][ 0]
        idx = valid[np.random.random_integers(0, high=np.shape(valid)[0])]
        pts2 = rotated[idx]

    # Rescale to actual size
    shape = shape[::-1]  # different convention [y, x]
    pts1 *= np.expand_dims(shape, axis=0).astype(np.float32)
    pts2 *= np.expand_dims(shape, axis=0).astype(np.float32)

    def ax(p, q): return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q): return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = np.transpose(np.stack(
        [[pts2[i][j] for i in range(4) for j in range(2)]], axis=0))
    homography = np.transpose(np.linalg.solve(a_mat, p_mat))
    return homography



def flat2mat(H):
    """
    Converts a flattened homography transformation with shape `[1, 8]` to its
    corresponding homography matrix with shape `[1, 3, 3]`.
    """
    return np.reshape(np.concatenate([H, np.ones([np.shape(H)[0], 1])], axis=1), [-1, 3, 3])


def invert_homography(H):
    """
    Computes the inverse transformation for a flattened homography transformation.
    """
    return mat2flat(np.linalg.inv((flat2mat(H))))

def mat2flat(H):
    """
    Converts an homography matrix with shape `[1, 3, 3]` to its corresponding flattened
    homography transformation with shape `[1, 8]`.
    """
    H = np.reshape(H, [-1, 9])
    return (H / H[:, 8:9])[:, :8]


def compute_valid_mask(image_shape, homography, erosion_radius=0):
    """
    Compute a boolean mask of the valid pixels resulting from an homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.
    Arguments:
        input_shape: Tensor of rank 2 representing the image shape, i.e. `[H, W]`.
        homography: Tensor of shape (B, 8) or (8,), where B is the batch size.
        erosion_radius: radius of the margin to be discarded.
    Returns: a Tensor of type `tf.int32` and shape (H, W).
    """
    #mask = H_transform(np.ones(image_shape), homography, interpolation='NEAREST')
    mask=cv2.warpPerspective(np.ones(image_shape),homography,image_shape)
    # if erosion_radius > 0:
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius*2,)*2)
    #     mask = tf.nn.erosion2d(
    #             mask[tf.newaxis, ..., tf.newaxis],
    #             tf.to_float(tf.constant(kernel)[..., tf.newaxis]),
    #             [1, 1, 1, 1], [1, 1, 1, 1], 'SAME')[0, ..., 0] + 1.
    # return tf.to_int32(mask)
    return None


def warp_points(points, homography):
    """
    Warp a list of points with the INVERSE of the given homography.
    The inverse is used to be coherent with tf.contrib.image.transform
    Arguments:
        points: list of N points, shape (N, 2).
        homography: batched or not (shapes (B, 8) and (8,) respectively).
    Returns: a Tensor of shape (N, 2) or (B, N, 2) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.
    """
    H = np.expand_dims(homography, axis=0) if len(homography.shape) == 1 else homography

    # Get the points to the homogeneous format
    num_points = np.shape(points)[0]
    points = np.cast(points, np.float32)[:, ::-1]
    points = np.concatenate([points, np.ones([num_points, 1]).astype(np.float32)], -1)

    # Apply the homography
    H_inv = np.transpose(flat2mat(invert_homography(H)))
    warped_points = np.tensordot(points, H_inv, [[1], [0]])
    warped_points = warped_points[:, :2, :] / warped_points[:, 2:, :]
    warped_points = np.transpose(warped_points, [2, 0, 1])[:, :, ::-1]

    return warped_points[0] if len(homography.shape) == 1 else warped_points


def filter_points(points, shape):
    mask = (points >= 0) & (points <= np.float(shape-1))
    return points.tensor( np.all(mask, -1))


# TODO: cleanup the two following functions
def warp_keypoints_to_list(packed_arg):
    """
    Warp a map of keypoints (pixel is 1 for a keypoint and 0 else) with
    the INVERSE of the homography H.
    The inverse is used to be coherent with tf.contrib.image.transform
    Arguments:
        packed_arg: a tuple equal to (keypoints_map, H)
    Returns: a Tensor of size (num_keypoints, 2) with the new coordinates
             of the warped keypoints.
    """
    keypoints_map = packed_arg[0]
    H = packed_arg[1]
    if len(H.shape.as_list()) < 2:
        H = np.expand_dims(H, 0)  # add a batch of 1
    # Get the keypoints list in homogeneous format
    keypoints = np.where(keypoints_map > 0).astype(np.float32)
    keypoints = keypoints[:, ::-1]
    n_keypoints = np.shape(keypoints)[0]
    keypoints = np.concatenate([keypoints, np.ones([n_keypoints, 1] ).astype(np.float32)], 1)

    # Apply the homography
    H_inv = invert_homography(H)
    H_inv = flat2mat(H_inv)
    H_inv = np.transpose(H_inv[0, ...])
    warped_keypoints = np.matmul(keypoints, H_inv)
    warped_keypoints = np.round(warped_keypoints[:, :2]
                                / warped_keypoints[:, 2:])
    warped_keypoints = warped_keypoints[:, ::-1]

    return warped_keypoints


def scatter_numpy(self, dim, index, src):
    """
    Writes all values from the Tensor src into self at the indices specified in the index Tensor.

    :param dim: The axis along which to index
    :param index: The indices of elements to scatter
    :param src: The source element(s) to scatter
    :return: self
    """
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    if self.ndim != index.ndim:
        raise ValueError("Index should have the same number of dimensions as output")
    if dim >= self.ndim or dim < -self.ndim:
        raise IndexError("dim is out of range")
    if dim < 0:
        # Not sure why scatter should accept dim < 0, but that is the behavior in PyTorch's scatter
        dim = self.ndim + dim
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = self.shape[:dim] + self.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and output should be the same size")
    if (index >= self.shape[dim]).any() or (index < 0).any():
        raise IndexError("The values of index must be between 0 and (self.shape[dim] -1)")

    def make_slice(arr, dim, i):
        slc = [slice(None)] * arr.ndim
        slc[dim] = i
        return slc

    # We use index and dim parameters to create idx
    # idx is in a form that can be used as a NumPy advanced index for scattering of src param. in self
    idx = [[*np.indices(idx_xsection_shape).reshape(index.ndim - 1, -1),
            index[make_slice(index, dim, i)].reshape(1, -1)[0]] for i in range(index.shape[dim])]
    idx = list(np.concatenate(idx, axis=1))
    idx.insert(dim, idx.pop())

    if not np.isscalar(src):
        if index.shape[dim] > src.shape[dim]:
            raise IndexError("Dimension " + str(dim) + "of index can not be bigger than that of src ")
        src_xsection_shape = src.shape[:dim] + src.shape[dim + 1:]
        if idx_xsection_shape != src_xsection_shape:
            raise ValueError("Except for dimension " +
                             str(dim) + ", all dimensions of index and src should be the same size")
        # src_idx is a NumPy advanced index for indexing of elements in the src
        src_idx = list(idx)
        src_idx.pop(dim)
        src_idx.insert(dim, np.repeat(np.arange(index.shape[dim]), np.prod(idx_xsection_shape)))
        self[idx] = src[src_idx]

    else:
        self[idx] = src

    return self

def warp_keypoints_to_map(packed_arg):
    """
    Warp a map of keypoints (pixel is 1 for a keypoint and 0 else) with
    the INVERSE of the homography H.
    The inverse is used to be coherent with tf.contrib.image.transform
    Arguments:
        packed_arg: a tuple equal to (keypoints_map, H)
    Returns: a map of keypoints of the same size as the original keypoint_map.
    """
    warped_keypoints = np.int32(warp_keypoints_to_list(packed_arg))
    n_keypoints = np.shape(warped_keypoints)[0]
    shape = np.shape(packed_arg[0])

    # Remove points outside the image
    zeros = np.cast(np.zeros([n_keypoints]), dtype=np.bool)
    ones = np.cast(np.ones([n_keypoints]), dtype=np.bool)
    loc = np.logical_and(np.where(warped_keypoints[:, 0] >= 0, ones, zeros),
                         np.where(warped_keypoints[:, 0] < shape[0],
                                  ones,
                                  zeros))
    loc = np.logical_and(loc, np.where(warped_keypoints[:, 1] >= 0, ones, zeros))
    loc = np.logical_and(loc,
                         np.where(warped_keypoints[:, 1] < shape[1],
                                  ones,
                                  zeros))

    warped_keypoints = warped_keypoints.tensor( loc)



    # Output the new map of keypoints
    # new_map = np.scatter_nd(warped_keypoints,
    #                         np.ones([np.shape(warped_keypoints)[0]], dtype=np.float32),
    #                         shape)
    new_map=None
    return new_map