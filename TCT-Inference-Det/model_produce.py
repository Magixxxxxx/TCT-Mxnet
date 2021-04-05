import os
import mxnet as mx
import sys

from utils.dataset import ImageTxtDataset
from mxnet import gluon, nd
from mxnet import autograd as ag

from mxnet import image
import mxnet as mx
from utils.backbone_complete import faster_rcnn_fpn_resnet50_v1b_coco

CLASSES = ['normal', 'ascus', 'asch', 'lsil', 'hsil_scc_omn', 'agc_adenocarcinoma_em',
           'vaginalis', 'monilia', 'dysbacteriosis_herpes_act', 'ec']

def export2save(params, symbol):

    from utils.load_net import load_model

    net = load_model(params, symbol)
    net.collect_params().reset_ctx(mx.cpu())
    net.hybridize()

    path = './IMAGES/10_6.jpg'
    im = image.imread(path)
    im = image.imresize(im, w=1164, h=800)
    im = mx.nd.image.to_tensor(im)
    im = mx.nd.image.normalize(im, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    im = mx.nd.expand_dims(im, axis=0)
    ip = mx.nd.array(im, mx.cpu())
    res = net(ip)

    net.save_parameters("{}_saved.params".format(model_params)) 

def save2export(params):
    net = faster_rcnn_fpn_resnet50_v1b_coco(classes=CLASSES, root='./models', pretrained_base=False,
                                                     per_device_batch_size=1)
    net.load_parameters(params, allow_missing=True, ignore_extra=True)

    net.collect_params().reset_ctx(mx.cpu(0))
    net.hybridize()

    from mxnet import image
    import mxnet as mx

    path = './IMAGES/10_6.jpg'
    im = image.imread(path)
    im = image.imresize(im, w=1164, h=800)
    im = mx.nd.image.to_tensor(im)
    im = mx.nd.image.normalize(im, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    im = mx.nd.expand_dims(im, axis=0)
    ip = mx.nd.array(im, mx.cpu(0))

    res = net(ip)
    
    net.export(path='./OUTPUT/', epoch=999)    

if __name__ == '__main__':
    from utils.config_val import Config
    config = Config()

    params = "MODEL/TCT-37.8.params"
    symbol = "MODEL/binary-cls-L-symbol.json"

    from utils.load_net import load_model

    net = load_model(params, symbol)
    net.load_parameters("MODEL/TCT-37.8.params")
    net.save_parameters("saved.params") 

    net2 = faster_rcnn_fpn_resnet50_v1b_coco(classes=CLASSES, root='./models', pretrained_base=False,
                                                     per_device_batch_size=1)
    net2.load_parameters("saved.params", allow_missing=True, ignore_extra=True)

    # if symbol:
    #     export2save(params,symbol)
    # else:
    #     save2export(params)