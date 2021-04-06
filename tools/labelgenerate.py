import datetime
import json
import os
import sys
import cv2
import imagesize
import numpy as np
import random

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

def coco_TCTInit():
    data_train = init_data_dict()
    data_test = init_data_dict()
    data_val = init_data_dict()

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
        -1: 1,
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
        5: 5,
        6: 6,
        8: 6,
        9: 6,
        10: 7,
        11: 8,
        12: 9,
        13: 9,
        14: 9,
        15: 5,
        16: 10
    }
    
    for class_name, id_ in classes_name_to_id.items():
        data_train['categories'].append(dict(
            supercategory=None,
            id=id_,
            name=class_name,
        ))
    data_test['categories'] = data_train['categories']
    data_val['categories'] = data_train['categories']

    return data_train, data_test, data_val, classes_name_to_id, id_reflect

def coco_getImgs(data, path):
    img_name_to_id = {}
    count_id = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            if ".jpg" in f:
                img = os.path.join(root, f)
                img_name_to_id[img] = count_id
                data['images'].append(dict(
                    license=0,
                    url=None,
                    file_name=img,
                    # height=int(img_data.shape[0]),
                    # width=int(img_data.shape[1]),
                    height=2048,
                    width=4096,
                    date_captured=None,
                    id=count_id,
                ))
                count_id += 1
                # if (int(img_data.shape[0]) != 2048 ) or (img_data.shape[1] != 4096):
                #     print(img_data.shape[1], img_data.shape[0])
    return data, img_name_to_id

def coco_addAnn(data, cat, x,y,w,h, img_id):
    data['annotations'].append(dict(
        id=len(data['annotations']),
        image_id=img_id,
        category_id=cat,
        segmentation=[[]],
        area=w * h,
        bbox=[x, y, w, h],
        iscrowd=0,
    ))

def main():
    datadir = "/root/commonfile/TCTAnnotatedData/HK_TCTAnnotated20200303/"
    data_train, data_test, data_val, classes_name_to_id, id_reflect = coco_TCTInit()
    data_train, img_name_to_id = coco_getImgs(data_train, datadir)
    data_test, _ = coco_getImgs(data_test, datadir)
    data_val, _ = coco_getImgs(data_val, datadir)

    data_set = {}
    cat_ids = set()

    for folder in os.listdir(datadir):
        curdir = datadir + folder
        labelfile = "{}/{}".format(curdir, os.listdir(curdir)[1])

        with open(labelfile) as lf:
            nums = lf.readline()
            for row in lf:
                row_data = row.split()
                cat = id_reflect[int(row_data[0])]
                pic = "{}/{}/DigitalSlice/OriginalImage/{}_{}.jpg".format(curdir, os.listdir(curdir)[0], row_data[1], row_data[2])
                x = int(row_data[3])
                y = int(row_data[4])
                w = int(row_data[5])
                h = int(row_data[6])

                if pic not in data_set:
                    data_set[pic] = []
                
                data_set[pic].append([cat,x,y,w,h])

                cat_ids.add(cat)
                div = random.randint(0,10)
                if div <= 8:
                    coco_addAnn(data_train, cat, x,y,w,h, img_name_to_id[pic])
                elif div == 9:
                    coco_addAnn(data_test, cat, x,y,w,h, img_name_to_id[pic])
                else:
                    coco_addAnn(data_val, cat, x,y,w,h, img_name_to_id[pic])

    print(data_set[pic])
    print(cat_ids)

    np.save('datasets',data_set)


classes_counter = {}

def func(keylist, read_dictionary, data, img_name_to_id):
    
    for k in keylist:
        for ann in read_dictionary[k]:
            coco_addAnn(data, *ann, img_name_to_id[k])

            classes_counter.setdefault(ann[0], 0)
            classes_counter[ann[0]] += 1
    print(classes_counter)
    

if __name__ == "__main__":
    read_dictionary = np.load('./zjw/datasets.npy', allow_pickle=True).item()
    keylist = list(read_dictionary.keys())
    random.shuffle(keylist)

    train_list = keylist[:1720]
    test_list = keylist[1720:1892]
    val_list = keylist[1892:]

    datadir = "/root/commonfile/TCTAnnotatedData/HK_TCTAnnotated20200303/"
    data_train, data_test, data_val, classes_name_to_id, id_reflect = coco_TCTInit()
    data_train, img_name_to_id = coco_getImgs(data_train, datadir)
    data_test['images'] = data_train['images']
    data_val['images'] = data_train['images']

    func(train_list, read_dictionary,  data_train, img_name_to_id)
    func(test_list, read_dictionary, data_test, img_name_to_id)
    func(val_list, read_dictionary, data_val, img_name_to_id)

    print(len(data_train['annotations']))
    print(len(data_test['annotations']))
    print(len(data_val['annotations']))

    with open("/root/userfolder/mxnet-fasterrcnn/zjw/train.json", 'w') as f:
        json.dump(data_train, f, indent=2)

    with open("/root/userfolder/mxnet-fasterrcnn/zjw/test.json", 'w') as f:
        json.dump(data_test, f, indent=2)

    with open("/root/userfolder/mxnet-fasterrcnn/zjw/val.json", 'w') as f:
        json.dump(data_val, f, indent=2)