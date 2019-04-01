from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import tarfile
import cv2
from tqdm import tqdm
from multiprocessing import Pool, Process, Manager, cpu_count
import h5py
import logging

import shutil
import torch

# Local includes
from base import BaseDataLoader
from data_loader.utils.pipeline import parse_primitives
from data_loader.utils.synthetic_shapes_funcs import set_random_state, generate_background
import data_loader.utils.synthetic_shapes_funcs as synthetic_dataset
import data_loader.utils.pipeline as pipeline

import torch.nn.functional as F


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


class SyntheticDataLoader(BaseDataLoader):
    def __init__(
            self, batch_size,
            shuffle, validation_split,
            num_workers, pin_memory=False, dataset_args={}
    ):
        self.dataset = SyntheticDataSet(**dataset_args)
        super(SyntheticDataLoader, self).__init__(
            self.dataset, batch_size, shuffle,
            validation_split, num_workers, pin_memory)


class SyntheticDataSet(Dataset):
    default_config = {
        'primitives': 'all',
        'truncate': {},
        'validation_size': -1,
        'test_size': -1,
        'on-the-fly': False,
        'cache_in_memory': False,
        'suffix': None,
        'add_augmentation_to_test_set': False,
        'num_parallel_calls': 10,
        'generation': {
            'split_sizes': {'training': 10000, 'validation': 200, 'test': 500},
            'image_size': [960, 1280],
            'random_seed': 0,
            'params': {
                'generate_background': {
                    'min_kernel_size': 150, 'max_kernel_size': 500,
                    'min_rad_ratio': 0.02, 'max_rad_ratio': 0.031},
                'draw_stripes': {'transform_params': (0.1, 0.1)},
                'draw_multiple_polygons': {'kernel_boundaries': (50, 100)}
            },
        },
        'preprocessing': {
            'resize': [240, 320],
            'blur_size': 11,
        },
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'valid_border_margin': 0,
            },
        }
    }
    drawing_primitives = [
        'draw_lines',
        'draw_polygon',
        'draw_multiple_polygons',
        'draw_ellipses',
        'draw_star',
        'draw_checkerboard',
        'draw_stripes',
        'draw_cube',
        'gaussian_noise'
    ]

    def __init__(self, data_dir, config=default_config, force_recache=False, shuffle=False, cache_in_memory=False):
        '''

        :param data_dir: Directory where you want to store the generated shapes
        :param config: configuration of the dataset(shapes,augmentations,...)
        :param force_recache: Force the dataset to regenerate the synthetic shapes
        :param shuffle: Boolean to enable reshuffling of the data
        :param cache_in_memory: Cache the whole dataset in memory rather then loading from disk
        '''
        self.data_dir = data_dir
        self.config = config
        self.useAlbumentations = False

        primitives = parse_primitives(config['primitives'], self.drawing_primitives)
        # if config['on-the-fly']:
        #     return None

        basepath = os.path.join(data_dir, 'synthetic_shapes')
        os.makedirs(basepath, exist_ok=True)

        self.train_samples = []
        for primitive in primitives:
            folder = os.path.join(basepath, primitive)
            img_folder = os.path.join(folder, 'images')
            pts_folder = os.path.join(folder, 'points')
            flag_generate_data = False

            # Check that we previosly created all the data
            if os.path.exists(folder):
                if os.path.exists(img_folder):
                    num_images = len(os.listdir(os.path.join(img_folder, 'training')))
                    num_pts = len(os.listdir(os.path.join(pts_folder, 'training')))
                    req_num = config['generation']['split_sizes']['training']
                    if num_images != req_num or num_pts != req_num:
                        flag_generate_data = True
                else:
                    flag_generate_data = True
            else:
                flag_generate_data = True
            if flag_generate_data or force_recache:
                self.generate_primitive_data(primitive, folder, config)

            img_files = os.listdir(os.path.join(img_folder, 'training'))
            pt_files = os.listdir(os.path.join(pts_folder, 'training'))
            # Gather filenames in all splits, optionally truncate
            img_files.sort()
            pt_files.sort()

            for idx in range(0, len(img_files)):
                sample = (os.path.join(img_folder, 'training', img_files[idx]),
                          os.path.join(pts_folder, 'training', pt_files[idx]))
                self.train_samples.append(sample)

        if shuffle:
            import random
            random.shuffle(self.train_samples)

        self.cache_in_memory = cache_in_memory
        if cache_in_memory:
            cached_samples = []
            for sample in tqdm(self.train_samples, desc="Loading Training Samples To Memory"):
                image = cv2.imread(sample[0], 0)
                image = image.astype('float32') / 255.

                pts = np.load(sample[1]).astype(np.float32)
                pts = np.reshape(pts, [-1, 2])

                data = {'image': image, 'keypoints': pts}
                data = pipeline.add_dummy_valid_mask(data)
                data = pipeline.add_keypoint_map(data)
                cached_samples.append(data)
            self.cached_samples = cached_samples

    def __len__(self):
        return len(self.train_samples)

    def __getitem__(self, index):
        def _read_image(filename):
            image = cv2.imread(filename, 0)
            image = image.astype('float32') / 255.
            return image

            # Python function

        def _read_points(filename):
            return np.load(filename).astype(np.float32)

        if not self.cache_in_memory:
            sample = self.train_samples[index]
            image = _read_image(sample[0])
            pts = np.reshape(_read_points(sample[1]), [-1, 2])

            data = {'image': image, 'keypoints': pts}
            data = pipeline.add_dummy_valid_mask(data)


            if self.config['add_augmentation_to_test_set']:
                if self.config['augmentation']['photometric']['enable']:
                    data=pipeline.photometric_augmentation(data,self.config['augmentation']['photometric'])
                if self.config['augmentation']['homographic']['enable']:
                    data=pipeline.homographic_augmentation(data,self.config['augmentation']['homographic'])

            # Apply augmentation
            # self.AugmentData(data)

            # Convert the point coordinates to a dense keypoint map
            data = pipeline.add_keypoint_map(data)
        else:
            data = self.cached_samples[index]
        # Convert to tensors
        image = torch.from_numpy(data['image'])
        valid_mask = torch.from_numpy(data['valid_mask'])
        keypoint_map = torch.from_numpy(data['keypoint_map'])

        # Have to pad the keypoint data if you use pythons default collate
        # keypoints=torch.from_numpy(data['keypoints'])

        image = torch.unsqueeze(image, 0)

        return image, keypoint_map, valid_mask

    def AugmentData(self, data):
        if self.useAlbumentations:
            import albumentations as A




        else:

            if self.config['augmentation']['photometric']['enable']:
                data = pipeline.photometric_augmentation(
                    data, **self.config['augmentation']['photometric'])
            if self.config['augmentation']['homographic']['enable']:
                data = pipeline.homographic_augmentation(
                    data, **self.config['augmentation']['homographic'])

        return data

    def dump_primitive_data_tar(self, primitive, tar_path, config):
        temp_dir = os.path.join(self.data_dir, 'tmp', primitive)

        set_random_state(np.random.RandomState(
            config['generation']['random_seed']))
        for split, size in self.config['generation']['split_sizes'].items():
            img_dir, pts_dir = [os.path.join(temp_dir, i, split) for i in ['images', 'points']]
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(pts_dir, exist_ok=True)

            num_processes = max(cpu_count() - 2, 1)
            indices = range(size)
            try:
                import parmap
                parmap.map(generate_pts, config, primitive, img_dir, pts_dir, indices, pm_pbar=True,
                           pm_processes=num_processes)
            except:
                for i in tqdm(indices):
                    generate_pts(config, primitive, img_dir, pts_dir, i)

        # Pack into a tar file
        tar = tarfile.open(tar_path, mode='w:gz')
        tar.add(temp_dir, arcname=primitive)
        tar.close()
        shutil.rmtree(temp_dir)

    def generate_primitive_data(self, primitive, folder, config):
        logging.info('Generating data for primitive {}'.format(primitive))
        set_random_state(np.random.RandomState(
            config['generation']['random_seed']))
        for split, size in self.config['generation']['split_sizes'].items():
            img_dir, pts_dir = [os.path.join(folder, i, split) for i in ['images', 'points']]
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(pts_dir, exist_ok=True)

            num_processes = max(cpu_count() - 2, 1)
            indices = range(size)
            # Don't know if this is actually faster
            try:
                import parmap
                parmap.map(generate_pts, config, primitive, img_dir, pts_dir, indices, pm_pbar=True,
                           pm_processes=num_processes, pm_chunksize=500)
            except:
                for i in tqdm(indices):
                    generate_pts(config, primitive, img_dir, pts_dir, i)


def generate_pts(config, primitive_type, img_dir, pts_dir, index):
    image = generate_background(
        config['generation']['image_size'],
        **config['generation']['params']['generate_background'])
    points = np.array(getattr(synthetic_dataset, primitive_type)(
        image, **config['generation']['params'].get(primitive_type, {})))
    points = np.flip(points, 1)  # reverse convention with opencv

    b = config['preprocessing']['blur_size']
    image = cv2.GaussianBlur(image, (b, b), 0)
    points = (points * np.array(config['preprocessing']['resize'], np.float)
              / np.array(config['generation']['image_size'], np.float))
    image = cv2.resize(image, tuple(config['preprocessing']['resize'][::-1]),
                       interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(str(os.path.join(img_dir, '{}.png'.format(index))), image)
    np.save(os.path.join(pts_dir, '{}.npy'.format(index)), points)


def generate_pts_no_save(config, primitive_type):
    image = generate_background(
        config['generation']['image_size'],
        **config['generation']['params']['generate_background'])
    points = np.array(getattr(synthetic_dataset, primitive_type)(
        image, **config['generation']['params'].get(primitive_type, {})))
    points = np.flip(points, 1)  # reverse convention with opencv

    b = config['preprocessing']['blur_size']
    image = cv2.GaussianBlur(image, (b, b), 0)
    points = (points * np.array(config['preprocessing']['resize'], np.float)
              / np.array(config['generation']['image_size'], np.float))
    image = cv2.resize(image, tuple(config['preprocessing']['resize'][::-1]),
                       interpolation=cv2.INTER_LINEAR)

    return image, points
