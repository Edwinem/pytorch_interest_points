from model.homographies import (sample_homography, compute_valid_mask,
                                             warp_points, filter_points)

import numpy as np

import cv2 as cv
import numpy as np



augmentations = [
        'additive_gaussian_noise',
        'additive_speckle_noise',
        'random_brightness',
        'random_contrast',
        'additive_shade',
        'motion_blur'
]


def additive_gaussian_noise(image, stddev_range=[5, 95]):
    stddev = np.random.uniform(0, *stddev_range)
    noise = np.random.randn(image.shape)*stddev
    #noisy_image = np.clip_by_value(image + noise, 0, 255)
    return noisy_image


def additive_speckle_noise(image, prob_range=[0.0, 0.005]):
    prob = tf.random_uniform((), *prob_range)
    sample = tf.random_uniform(tf.shape(image))
    noisy_image = tf.where(sample <= prob, tf.zeros_like(image), image)
    noisy_image = tf.where(sample >= (1. - prob), 255.*tf.ones_like(image), noisy_image)
    return noisy_image


def random_brightness(image, max_abs_change=50):
    return tf.clip_by_value(tf.image.random_brightness(image, max_abs_change), 0, 255)


def random_contrast(image, strength_range=[0.5, 1.5]):
    return tf.clip_by_value(tf.image.random_contrast(image, *strength_range), 0, 255)


def additive_shade(image, nb_ellipses=20, transparency_range=[-0.5, 0.8],
                   kernel_size_range=[250, 350]):

    def _py_additive_shade(img):
        min_dim = min(img.shape[:2]) / 4
        mask = np.zeros(img.shape[:2], np.uint8)
        for i in range(nb_ellipses):
            ax = int(max(np.random.rand() * min_dim, min_dim / 5))
            ay = int(max(np.random.rand() * min_dim, min_dim / 5))
            max_rad = max(ax, ay)
            x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center
            y = np.random.randint(max_rad, img.shape[0] - max_rad)
            angle = np.random.rand() * 90
            cv.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

        transparency = np.random.uniform(*transparency_range)
        kernel_size = np.random.randint(*kernel_size_range)
        if (kernel_size % 2) == 0:  # kernel_size has to be odd
            kernel_size += 1
        mask = cv.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
        shaded = img * (1 - transparency * mask[..., np.newaxis]/255.)
        return np.clip(shaded, 0, 255)

    shaded = tf.py_func(_py_additive_shade, [image], tf.float32)
    res = tf.reshape(shaded, tf.shape(image))
    return res


def motion_blur(image, max_kernel_size=10):

    def _py_motion_blur(img):
        # Either vertial, hozirontal or diagonal blur
        mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
        ksize = np.random.randint(0, (max_kernel_size+1)/2)*2 + 1  # make sure is odd
        center = int((ksize-1)/2)
        kernel = np.zeros((ksize, ksize))
        if mode == 'h':
            kernel[center, :] = 1.
        elif mode == 'v':
            kernel[:, center] = 1.
        elif mode == 'diag_down':
            kernel = np.eye(ksize)
        elif mode == 'diag_up':
            kernel = np.flip(np.eye(ksize), 0)
        var = ksize * ksize / 16.
        grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
        gaussian = np.exp(-(np.square(grid-center)+np.square(grid.T-center))/(2.*var))
        kernel *= gaussian
        kernel /= np.sum(kernel)
        img = cv.filter2D(img, -1, kernel)
        return img

    blurred = tf.py_func(_py_motion_blur, [image], tf.float32)
    return tf.reshape(blurred, tf.shape(image))


def parse_primitives(names, all_primitives):
    p = all_primitives if (names == 'all') \
            else (names if isinstance(names, list) else [names])
    assert set(p) <= set(all_primitives)
    return p


def photometric_augmentation(data, **config):
    primitives = parse_primitives(config['primitives'], photaug.augmentations)
    prim_configs = [config['params'].get(
                         p, {}) for p in primitives]

    indices = np.arange(len(primitives))
    if config['random_order']:
        indices = tf.random_shuffle(indices)

    def step(i, image):
        fn_pairs = [(tf.equal(indices[i], j),
                     lambda p=p, c=c: getattr(photaug, p)(image, **c))
                    for j, (p, c) in enumerate(zip(primitives, prim_configs))]
        image = tf.case(fn_pairs)
        return i + 1, image

    _, image = tf.while_loop(lambda i, image: tf.less(i, len(primitives)),
                             step, [0, data['image']], parallel_iterations=1)

    return {**data, 'image': image}


def homographic_augmentation(data, add_homography=False, **config):
    image_shape = np.shape(data['image'])[:2]
    homography = sample_homography(image_shape, **config['params'])[0]
    warped_image = tf.contrib.image.transform(
            data['image'], homography, interpolation='BILINEAR')
    valid_mask = compute_valid_mask(image_shape, homography,
                                    config['valid_border_margin'])

    warped_points = warp_points(data['keypoints'], homography)
    warped_points = filter_points(warped_points, image_shape)

    ret = {**data, 'image': warped_image, 'keypoints': warped_points,
           'valid_mask': valid_mask}
    if add_homography:
        ret['homography'] = homography
    return ret


def add_dummy_valid_mask(data):
    valid_mask = np.ones(np.shape(data['image'])[:2], dtype=np.int32)
    return {**data, 'valid_mask': valid_mask}


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


def add_keypoint_map(data):
    image_shape = np.shape(data['image'])
    kp = np.floor(data['keypoints']).astype(np.int32)
    kmap = np.zeros(image_shape).astype(np.float32)
    for p in kp:
        kmap[p[0],p[1]]=1.0
    return {**data, 'keypoint_map': kmap}


def downsample(image, coordinates, **config):
    with tf.name_scope('gaussian_blur'):
        k_size = config['blur_size']
        kernel = cv.getGaussianKernel(k_size, 0)[:, 0]
        kernel = np.outer(kernel, kernel).astype(np.float32)
        kernel = tf.reshape(tf.convert_to_tensor(kernel), [k_size]*2+[1, 1])
        pad_size = int(k_size/2)
        image = tf.pad(image, [[pad_size]*2, [pad_size]*2, [0, 0]], 'REFLECT')
        image = tf.expand_dims(image, axis=0)  # add batch dim
        image = tf.nn.depthwise_conv2d(image, kernel, [1, 1, 1, 1], 'VALID')[0]

    with tf.name_scope('downsample'):
        ratio = tf.divide(tf.convert_to_tensor(config['resize']), tf.shape(image)[0:2])
        coordinates = coordinates * tf.cast(ratio, tf.float32)
        image = tf.image.resize_images(image, config['resize'],
                                       method=tf.image.ResizeMethod.BILINEAR)

    return image, coordinates


def ratio_preserving_resize(image, **config):
    target_size = tf.convert_to_tensor(config['resize'])
    scales = tf.to_float(tf.divide(target_size, tf.shape(image)[:2]))
    new_size = tf.to_float(tf.shape(image)[:2]) * tf.reduce_max(scales)
    image = tf.image.resize_images(image, tf.to_int32(new_size),
                                   method=tf.image.ResizeMethod.BILINEAR)
    return tf.image.resize_image_with_crop_or_pad(image, target_size[0], target_size[1])