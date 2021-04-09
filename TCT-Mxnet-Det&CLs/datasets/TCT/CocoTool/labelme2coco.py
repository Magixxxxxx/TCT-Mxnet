import datetime
import json
import os
import os.path as osp
import sys
import cv2

import imagesize

try:
    import pycocotools.mask
except ImportError:
    print('Please install pycocotools:\n\n    pip install pycocotools\n')
    sys.exit(1)

def init_data_dict():
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=None,
            contributor=None,
            date_created=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
        licenses=[dict(
            url=None,
            id=0,
            name=None,
        )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type='detection',
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )
    return data


def main(output_dir='./json', ann_file='train.txt', op_name='train.json'):
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    now = datetime.datetime.now()
    data = init_data_dict()

    classes_name_to_id = {
        "normal": 1,
        "ascus": 2,
        "asch": 3,
        "lsil": 4,
        "hsil_scc_omn": 5,
        "agc_adenocarcinoma_em": 6,
        "vaginalis": 7,
        "monilia": 8,
        "dysbacteriosis_herpes_act": 9,
        "ec": 10
    }

    id_reflect = {
        -1: 1, 0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5, 5: 5,
        6: 6, 8: 6, 9: 6,
        10: 7,
        11: 8,
        12: 9, 13: 9, 14: 9,
        15: 5,
        16: 10,
        21: 1, 22:1, 23:1 ,24:1, 26:1
    }

    for class_name, id_ in classes_name_to_id.items():
        data['categories'].append(dict(
            supercategory=None,
            id=id_,
            name=class_name,
        ))

    # print(data['categories'])
    # ---------------- make categories e-------------

    out_ann_file = osp.join(output_dir, op_name)

    count_id = 0
    # for i, filename in enumerate(os.listdir(ann_dir)):
    with open(ann_file, 'r', encoding='utf8') as f:
        for idx_, example in enumerate(f):
            example = example.strip()
            print('%d----->%s' % (idx_, example))
            # Loop images
            if '.txt' in example and os.path.exists(example[:-4] + '.jpg'):
                im_path = example[:-4] + '.jpg'
                an_path = example
                # im = cv2.imread(im_path)
                width_, height_ = imagesize.get(im_path)
                # ------------data['annotations'] s---------------
                flag = False
                with open(an_path, 'r', encoding='utf8') as ann_f:
                    ann_f.readline()
                    for i, line in enumerate(ann_f.readlines()):
                        line = line.strip()
                        value_list = line.split()
                        _type = int(value_list[0])
                        if _type in id_reflect.keys():
                            # whether own objects
                            if not flag:
                                flag = True

                            x_lt = int(value_list[1]) if int(value_list[1]) > 0 else 1
                            y_lt = int(value_list[2]) if int(value_list[2]) > 0 else 1
                            w = int(value_list[3]) if int(value_list[1]) > 0 else int(value_list[3]) - (
                                    1 - int(value_list[1]))
                            h = int(value_list[4]) if int(value_list[2]) > 0 else int(value_list[4]) - (
                                    1 - int(value_list[2]))
                            cls_id = id_reflect[_type]

                            data['annotations'].append(dict(
                                id=len(data['annotations']),
                                image_id=count_id,
                                category_id=cls_id,
                                segmentation=[[]],
                                area=w * h,
                                bbox=[x_lt, y_lt, w, h],
                                iscrowd=0,
                            ))

                # ------------data['annotations'] s---------------

                if flag:
                    
                    # ------------data['images'] s---------------
                    data['images'].append(dict(
                        license=0,
                        url=None,
                        file_name=im_path,
                        height=int(height_),
                        width=int(width_),
                        date_captured=None,
                        id=count_id,
                    ))
                    
                    count_id += 1
                    # ------------data['images'] e---------------

    with open(out_ann_file, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == '__main__':
    ann_file = sys.argv[1]
    op_name = sys.argv[2]
    main(output_dir='./json', ann_file=ann_file, op_name=op_name)
    print(sys.argv[1] + " done!")
