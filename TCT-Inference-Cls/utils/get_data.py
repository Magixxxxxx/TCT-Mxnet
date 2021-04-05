import mxnet as mx

from gluoncv.data.transforms import bbox as tbbox
from gluoncv.data.transforms import image as timage


class DATATransformer(object):
    def __init__(self, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), short=800, max_size=1333):
        self._mean = mean
        self._std = std
        self._short = short
        self._max_size = max_size

    def __call__(self, im, label=None):
        h, w = im.shape[0], im.shape[1]
        # im = im / 255.0

        im = timage.resize_short_within(mx.nd.array(im), self._short, self._max_size, interp=1)
        # no scaling ground-truth, return image scaling ratio instead
        # label , np.ndarray ---> (N, 4 +)
        bbox = None
        if label is not None:
            bbox = tbbox.resize(label, (w, h), (im.shape[1], im.shape[0]))
        im_scale = h / float(im.shape[0])
        # convert [0-255] ---> [0, 1]
        im = mx.nd.image.to_tensor(im)

        img_np = im.asnumpy()
        self.mean = tuple(img_np.mean(axis = (1,2)))
        self.std = tuple(img_np.std(axis = (1,2)))
        print('transform ',self.std)

        im = mx.nd.image.normalize(im, mean=self._mean, std=self._std)
        return im, bbox, im_scale


class DATATransformer_no_label(object):
    def __init__(self, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), short=800, max_size=1333):
        self._mean = mean
        self._std = std
        self._short = short
        self._max_size = max_size

    def __call__(self, im):
        h, w = im.shape[0], im.shape[1]
        # im = im / 255.0

        # im = timage.resize_short_within(mx.nd.array(im), self._short, self._max_size, interp=1)
        # no scaling ground-truth, return image scaling ratio instead
        # label , np.ndarray ---> (N, 4 +)
        im = mx.nd.array(im)
        im_scale = h / float(im.shape[0])
        # convert [0-255] ---> [0, 1]
        im = mx.nd.image.to_tensor(im)
        im = mx.nd.image.normalize(im, mean=self._mean, std=self._std)
        return im, im_scale
