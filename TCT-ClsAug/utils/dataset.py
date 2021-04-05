from mxnet.gluon.data.vision.datasets import dataset
from mxnet import gluon, image, init, nd
import numpy as np
import os


class ImageTxtDataset(dataset.Dataset):
    def __init__(self, root, flag=1, transform=None):
        print('loading dataset...')
        self._root = os.path.expanduser(root)
        self._flag = flag
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self._list_images(self._root)

    def _list_images(self, root):
        self.items = []
        with open(root, 'r') as txt_f:
            for line in txt_f:
                line = line.strip()
                self.items.append([line[:-2], line[-1]])

    def __getitem__(self, idx):
        img = image.imread(os.path.join('/root/commonfile/TCTAnnotatedData', self.items[idx][0]), self._flag)
        label = np.float32(self.items[idx][1])

        if self._transform is not None:
            return self._transform(img), label
        return img, label

    def __len__(self):
        return len(self.items)
