import mxnet as mx
import numpy as np
import os
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import image as timage

class myTrainTransform(object):
    def __init__(self, mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225), **kwargs):
        print('normalize each img')
        self._mean = mean
        self._std = std

    def __call__(self, src):
        # h, w = src.shape[0], src.shape[1]
        # img = timage.resize_short_within(nd.array(src), short=self._short,
        #                                  max_size=self._max_size, interp=1)
        img = timage.imresize(mx.nd.array(src), w=1164, h=800, interp=1)
        img = mx.nd.image.to_tensor(img)
        img, _ = timage.random_flip(img)

        img_np = img.asnumpy()
        self._mean = tuple(img_np.mean(axis = (1,2)))
        self._std = tuple(img_np.std(axis = (1,2)))

        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img
        #return img, label.astype(img.dtype)

class myTestTransform(object):
    def __init__(self, mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225), **kwargs):
        self._mean = mean
        self._std = std

    def __call__(self, src):
        img = timage.imresize(mx.nd.array(src), w=1164, h=800, interp=1)
        img = mx.nd.image.to_tensor(img)

        img_np = img.asnumpy()
        self._mean = tuple(img_np.mean(axis = (1,2)))
        self._std = tuple(img_np.std(axis = (1,2)))

        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img


class Config(object):
    classes = 1
    epochs = 2
    lr = 0.001

    momentum = 0.9
    wd = 0.0001

    lr_factor = 0.1
    lr_steps = [1, 2, np.inf]

    num_workers = 2
    ctx = [mx.gpu(i) for i in range(2)]

    num_gpus = len(ctx)

    jitter_param = 0.4
    lighting_param = 0.1

    path = './datasets'

    train_path = os.path.join(path, 'train.txt')
    val_path = os.path.join(path, 'test.txt')
    test_path = os.path.join(path, 'test.txt')

    def __init__(self):
        self.per_device_batch_size = 3
        self.batch_size = self.per_device_batch_size * max(self.num_gpus, 1)
        self.transform_train = myTrainTransform()
        self.transform_test = myTestTransform()

        # self.transform_train = transforms.Compose([
        #     transforms.Resize(size=(1164, 800)),
        #     transforms.RandomFlipLeftRight(),
        #     #transforms.RandomColorJitter(brightness=self.jitter_param, contrast=self.jitter_param,saturation=self.jitter_param),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0, 0, 0], [1, 1, 1])
        #     #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])

        # self.transform_test = transforms.Compose([
        #     transforms.Resize(size=(1164, 800)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0, 0, 0], [1, 1, 1])
        #     #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
