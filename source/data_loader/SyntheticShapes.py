from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import tarfile
import cv2
from tqdm import tqdm
from multiprocessing import Pool, Process, Manager,cpu_count

import shutil


from base import BaseDataLoader

from data_loader.utils.pipeline import parse_primitives
from data_loader.utils.synthetic_shapes_funcs import set_random_state,generate_background
import data_loader.utils.synthetic_shapes_funcs as synthetic_dataset


class SyntheticDataLoader(BaseDataLoader):
    def __init__(
        self, batch_size,
        shuffle, validation_split,
        num_workers,pin_memory=False, dataset_args={}
    ):
        self.dataset = SyntheticDataSet(**dataset_args)
        super(SyntheticDataLoader, self).__init__(
            self.dataset, batch_size, shuffle,
            validation_split, num_workers,pin_memory)


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

    def __init__(self, data_dir, config=default_config):
        self.data_dir=data_dir
        self.config=config


        primitives = parse_primitives(config['primitives'], self.drawing_primitives)
        if config['on-the-fly']:
            return None

        basepath=os.path.join(data_dir,'synthetic_shapes')
        os.makedirs(basepath,exist_ok=True)



        splits = {s: {'images': [], 'points': []}
                  for s in ['training', 'validation', 'test']}
        for primitive in primitives:
            tar_path = os.path.join(basepath, '{}.tar.gz'.format(primitive))
            if not os.path.exists(tar_path):
                self.dump_primitive_data(primitive, tar_path, config)

            # Untar locally
            tmp_dir = os.path.join(data_dir, 'tmp')
            os.makedirs(tmp_dir, exist_ok=True)
            tar = tarfile.open(tar_path)
            tar.extractall(path=tmp_dir)
            tar.close()

            # Gather filenames in all splits, optionally truncate
            truncate = config['truncate'].get(primitive, 1)
            path = os.path.join(tmp_dir, primitive)
            for s in splits:
                dir=os.path.join(path, 'images', s)
                e = [str(p) for p in os.listdir(dir)]
                f = [p.replace('images', 'points') for p in e]
                f = [p.replace('.png', '.npy') for p in f]
                splits[s]['images'].extend(e[:int(truncate * len(e))])
                splits[s]['points'].extend(f[:int(truncate * len(f))])

        # Shuffle
        for s in splits:
            perm = np.random.RandomState(0).permutation(len(splits[s]['images']))
            for obj in ['images', 'points']:
                splits[s][obj] = np.array(splits[s][obj])[perm].tolist()

        self.split= splits


    def __getitem__(self, index):
        frame, label = self.dataset.__getitem__(index)
        data = {
            "frame": frame,
            "label": label,
        }
        return data

    def __len__(self):
        return len(self.dataset)

    def dump_primitive_data(self, primitive, tar_path, config):
        temp_dir =os.path.join(self.data_dir, 'tmp', primitive)

        set_random_state(np.random.RandomState(
                config['generation']['random_seed']))
        for split, size in self.config['generation']['split_sizes'].items():
            img_dir, pts_dir = [os.path.join(temp_dir, i, split) for i in ['images', 'points']]
            os.makedirs(img_dir,exist_ok=True)
            os.makedirs(pts_dir,exist_ok=True)

            num_processes=max(cpu_count()-2,1)
            indices=range(size)
            try:
                import parmap
                parmap.map(generate_pts, config,primitive,img_dir,pts_dir,indices, pm_pbar=True,pm_processes=num_processes)
            except:
                for i in tqdm(indices):
                     generate_pts(config,primitive,img_dir,pts_dir,i)

        # Pack into a tar file
        tar = tarfile.open(tar_path, mode='w:gz')
        tar.add(temp_dir, arcname=primitive)
        tar.close()
        shutil.rmtree(temp_dir)



def generate_pts(config,primitive_type,img_dir,pts_dir,index):
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
