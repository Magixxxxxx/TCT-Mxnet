from glob import glob
from utils.get_data import DATATransformer
import cv2
from utils.load_gt import load_gt_txt
from utils.load_net import load_model
from utils.vis import draw_box
import numpy as np

def ret2np(ret):
    cat_ids = ret[0].squeeze().asnumpy()  # (10000, )

    index_ = np.where(cat_ids > -1)
    cat_ids = cat_ids[index_]

    cat_scores = ret[1].squeeze().asnumpy()  # (10000, )
    cat_scores = cat_scores[index_]

    bboxes = ret[2].squeeze().asnumpy()  # (10000, 4)
    bboxes = bboxes[index_]

    bicls_scores = ret[3].squeeze().asnumpy()  # (1, )

    label_infer = np.concatenate((bboxes, np.expand_dims(cat_ids, axis=1),
                                  np.expand_dims(cat_scores, axis=1)), axis=1).tolist()
    return label_infer

if __name__ == '__main__':
    transformer = DATATransformer()
    model_prams = 'MODEL/TCT-37.8.params'
    model_symbol  = 'MODEL/binary-cls-L-symbol.json'
    img_folder = 'IMAGES/*.jpg'
    out_folder = 'OUTPUT-37.8/'

    net = load_model(model_prams, model_symbol)

    for im_path in glob(img_folder):
        print("inferring {}......".format(im_path))

        im = cv2.imread(im_path)

        im_dir,_,im_name = im_path.rpartition('\\')
        ann_path = im_name.split('.')[0] + '.txt'

        label_gt = load_gt_txt(im_dir+'/'+ann_path)

        # bbox ---> list: [(x1, y1, x2, y2, cat_id)]
        # ret--->len=4 list
        # ret[0]-->cat_id(1, 10000, 1), ret[1]-->score(1, 10000, 1),
        # ret[2]-->bbox(1, 10000, 4), ret[3]-->bicls(1,1)
        x, bbox, im_scale = transformer(im, label_gt)
        ret = net(x.expand_dims(axis=0))
        label_infer = ret2np(ret)

        print(label_infer)
        draw_box(im, labels_infer=label_infer, labels_gt=bbox,
                 im_name=im_name, im_scale=im_scale, start_clsid=0, save_prefix=out_folder)
