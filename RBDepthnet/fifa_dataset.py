import os
import random

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as tf


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None):
        self.args = args
        self.filenames = []
        if mode == 'train':
            with open(os.path.join(args.dataset_root, 'train.txt'), 'r') as f:
                for line in f.readlines():
                    self.filenames.append(line)
        else:
            with open(os.path.join(args.dataset_root, 'test.txt'), 'r') as f:
                for line in f.readlines():
                    self.filenames.append(line)
        self.mode = mode
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1, 1, 1))
        ]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, item):
        sample_path = self.filenames[item]
        image_path = sample_path.strip().split()[0]
        depth_path = sample_path.strip().split()[1]
        image = Image.open(image_path)
        depth = Image.open(depth_path)
        image = image.resize((704,396), Image.ANTIALIAS).transpose(Image.FLIP_TOP_BOTTOM)
        depth = depth.resize((704,396), Image.ANTIALIAS).transpose(Image.FLIP_TOP_BOTTOM)

        if self.mode == 'train':
            #水平翻转
            if np.random.rand() > 0.5:
                image, depth = tf.hflip(image), tf.hflip(depth)
            #随机裁剪
            if np.random.rand() > 0.5:
                image = np.asarray(image, dtype=np.float32)
                depth = np.asarray(depth, dtype=np.float32)
                image, depth = self.random_crop(image, depth,
                                                self.args.input_height,
                                                self.args.input_width)

            else:
                image = image.resize((self.args.input_width, self.args.input_height),
                                     Image.ANTIALIAS)
                depth = depth.resize((self.args.input_width, self.args.input_height),
                                     Image.ANTIALIAS)
                image = np.asarray(image, dtype=np.float32)
                depth = np.asarray(depth, dtype=np.float32)
            #随机图像增强
            if np.random.rand() > 0.5:
                image /= 255.0
                image = self.random_augment(image)
                image *= 255.0
            #np 转 PIL
            image = Image.fromarray(np.uint8(image)).convert('RGB')
            depth = Image.fromarray(np.uint8(depth))
            #归一化处理
            image = self.transform(image)
            depth = tf.to_tensor(depth)
        else:
            image = image.resize((self.args.input_width, self.args.input_height),
                                 Image.ANTIALIAS)
            depth = depth.resize((self.args.input_width, self.args.input_height),
                                 Image.ANTIALIAS)
            image = self.transform(image)
            depth = torch.from_numpy(np.asarray(depth, dtype=np.float32))
        print(image.shape, depth.shape)
        print(image)
        # print(depth)
        return {'src': image, 'depth': depth,
                'image_path': image_path, 'depth_path': depth_path}
    def random_crop(self, image, depth, height, width):
        assert image.shape[0] >= height
        assert image.shape[1] >= width
        assert depth.shape[0] == image.shape[0]
        assert depth.shape[1] == image.shape[1]

        x = random.randint(0, image.shape[1] - width)
        y = random.randint(0, image.shape[0] - height)

        image = image[y : y + height, x : x + width, :]
        depth = depth[y : y + height, x : x + width, :]

        return image, depth

    def random_augment(self, image):
        #gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        if np.random.rand() > 0.5:
            image = image ** gamma

        #brightness augmentation
        brightness = random.uniform(0.9, 1.1)
        if np.random.rand() > 0.5:
            image = image * brightness

        #color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        colorImage = np.stack([white * colors[i] for i in range(3)], axis=2)
        if np.random.rand() > 0.5:
            image *= colorImage
            image = np.clip(image, 0, 1)

        return image

class FifaDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode)
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)
        else:
            self.testing_samples = DataLoadPreprocess(args, mode)
            self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                               shuffle=False,
                               num_workers=1,
                               pin_memory=False,
                               sampler=self.eval_sampler)