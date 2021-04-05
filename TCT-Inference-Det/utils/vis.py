import cv2
import os.path as osp
import os
import copy


def draw_box(im, labels_infer, labels_gt, im_name, im_scale=1, start_clsid=0, save_prefix='./OUTPUT-0999/'):
    """

    :param im:
    :param labels: [(x1, y1, x2, y2, cat_id, score), ...]
    :param labels_gt: [(x1, y1, x2, y2, cat_id), ...]
    :param save_prefix:
    :return:
    """
    if not osp.exists(save_prefix):
        os.makedirs(save_prefix)

    id2info = {
        0: ['normal', (255, 0, 0)], 1: ['ascus', (147, 20, 255)], 2: ['asch', (127, 255, 0)],
        3: ['lsil', (50, 205, 154)], 4: ['hsil', (0, 255, 255)], 5: ['agc', (0, 140, 255)],
        6: ['adenocarcinoma', (0, 69, 255)], 7: ['vaginalis', (0, 0, 255)], 8: ['monilia', (0, 0, 128)],
        9: ['dysbacteriosis', (0, 252, 124)]
    } if start_clsid == 0 else {
        1: ['normal', (255, 0, 0)], 2: ['ascus', (147, 20, 255)], 3: ['asch', (127, 255, 0)],
        4: ['lsil', (50, 205, 154)], 5: ['hsil', (0, 255, 255)], 6: ['agc', (0, 140, 255)],
        7: ['adenocarcinoma', (0, 69, 255)], 8: ['vaginalis', (0, 0, 255)], 9: ['monilia', (0, 0, 128)],
        10: ['dysbacteriosis', (0, 252, 124)]
    }

    id2info_txt = {
        1: ['normal', (255, 0, 0)], 2: ['ascus', (147, 20, 255)], 3: ['asch', (127, 255, 0)],
        4: ['lsil', (50, 205, 154)], 5: ['hsil', (0, 255, 255)], 6: ['scc', (0, 140, 255)],
        7: ['agc', (0, 69, 255)], 8: ['ais', (0, 0, 255)], 9: ['adenocarcinoma', (0, 0, 128)],
        10: ['em', (0, 252, 124)], 11: ['vaginalia', (0, 252, 124)], 12: ['monilia', (0, 252, 124)],
        13: ['dysbacteriosis', (0, 252, 124)], 14: ['herpes', (0, 252, 124)],
        15: ['actinomyces', (0, 252, 124)],
        16: ['omn', (0, 252, 124)], 17: [['ec', (0, 252, 124)]]}

    im_inf = copy.deepcopy(im)

    for label_inf in labels_infer:
        x1, y1, x2, y2, cat_id, score = label_inf
        if cat_id==0.0:
            continue
        cv2.rectangle(im_inf, (int(x1 * im_scale), int(y1 * im_scale)),
                      (int(x2 * im_scale), int(y2 * im_scale)), color=id2info[int(cat_id)][1], thickness=2)
        cv2.putText(im_inf, text='{}-{:.3f}'.format(id2info[int(cat_id)][0], score),
                    org=(int(x1 * im_scale), int(y1 * im_scale - 5)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=id2info[int(cat_id)][1], thickness=2)
    print(osp.join(save_prefix, im_name))
    cv2.imwrite(osp.join(save_prefix, im_name), im_inf)

    for label_f in labels_gt:
        x1, y1, x2, y2, cat_id = label_f
        cv2.rectangle(im, (int(x1 * im_scale), int(y1 * im_scale)),
                      (int(x2 * im_scale), int(y2 * im_scale)), color=id2info_txt[int(cat_id) + 1][1], thickness=2)
        cv2.putText(im, text='{}-{}'.format(id2info_txt[int(cat_id) + 1][0], 'gt'),
                    org=(int(x1 * im_scale), int(y1 * im_scale - 3)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=id2info_txt[int(cat_id) + 1][1], thickness=2)
    cv2.imwrite(osp.join(save_prefix, im_name.split('.')[0] + 'gt.jpg'), im)
