import numpy as np
import xml.etree.ElementTree as ET

name2id = {'normal': 1, 'ascus': 2, 'asch': 3, 'lsil': 4, 'hsil': 5, 'agc': 6,
           'adenocarcinoma': 7, 'vaginalis': 8, 'monilia': 9, 'dysbacteriosis': 10}


def load_gt_txt(file_path):
    res = []
    with open(file_path, 'r') as f:
        f.readline()
        for line in f:
            lt = line.split()
            print(lt)
            cat_id = int(lt[0])
            x1 = int(lt[1])
            y1 = int(lt[2])
            w = int(lt[3])
            h = int(lt[4])
            res.append((x1, y1, x1 + w, y1 + h, cat_id))
    return np.array(res)


def load_gt_xml(xmlf):
    """
    parse xml file
    :param xmlf: xml file path
    :return: [(x1, y1, x2, y2, cat_id),(),...]
    """
    tree = ET.parse(xmlf)
    root = tree.getroot()

    objs = root.findall('object')

    res = []
    # find box from xml file
    for i, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        cls = obj.find('name').text.lower().strip()
        res.append((x1, y1, x2, y2, name2id[cls]))

    return res
