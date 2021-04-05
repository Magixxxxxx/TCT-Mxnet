import mxnet as mx
import numpy as np
import os
from mxnet.gluon.data.vision import transforms

class myNormalize(transforms.Normalize):
    def hybrid_forward(self, F, x):
        if is_np_array():
            F = F.npx

        img = x.asnumpy()
        self._mean = img.mean(axis = (1,2))
        self._std = img.std(axis = (1,2))
        print(self._mean)
        return F.image.normalize(x, self._mean, self._std)

class Config(object):
    classes = 1
    epochs = 2
    lr = 0.001

    momentum = 0.9
    wd = 0.0001

    lr_factor = 0.1
    lr_steps = [1, 2, np.inf]

    num_workers = 0
    ctx = [mx.gpu(i) for i in range(8)]

    num_gpus = len(ctx)
    print(num_workers,num_gpus,ctx)

    jitter_param = 0.4
    lighting_param = 0.1

    path = './datasets'

    train_path = os.path.join(path, 'train.txt')
    val_path = os.path.join(path, 'test.txt')
    test_path = os.path.join(path, 'test.txt')

    def __init__(self):
        self.per_device_batch_size = 1
        self.batch_size = self.per_device_batch_size * max(self.num_gpus, 1)

        self.transform_train = transforms.Compose([
            transforms.Resize(size=(1164, 800)),
            transforms.RandomFlipLeftRight(),
            transforms.RandomColorJitter(brightness=self.jitter_param, contrast=self.jitter_param,
                                         saturation=self.jitter_param),
            transforms.ToTensor(),
            myNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize(size=(1164, 800)),
            transforms.ToTensor(),
            myNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
