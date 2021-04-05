import mxnet as mx
import numpy as np
import os
from mxnet.gluon.data.vision import transforms


class Config(object):
    classes = 1
    epochs = 2
    lr = 0.001

    momentum = 0.9
    wd = 0.0001

    lr_factor = 0.1
    lr_steps = [1, 2, np.inf]

    num_gpus = 8
    num_workers = 0
    ctx = [mx.gpu(i) for i in range(8)]
    print('val')

    jitter_param = 0.4
    lighting_param = 0.1

    path = './datasets'

    train_path = os.path.join(path, 'mini.txt')
    val_path = os.path.join(path, 'mini.txt')
    test_path = os.path.join(path, 'mini.txt')

    def __init__(self):
        self.per_device_batch_size = 1
        self.batch_size = self.per_device_batch_size * max(self.num_gpus, 1)

        self.transform_train = transforms.Compose([
            transforms.Resize(size=(1164, 800)),
            transforms.RandomFlipLeftRight(),
            transforms.RandomColorJitter(brightness=self.jitter_param, contrast=self.jitter_param,
                                         saturation=self.jitter_param),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize(size=(1164, 800)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
