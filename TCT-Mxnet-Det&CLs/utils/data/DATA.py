from gluoncv import data as gdata
import os
import numpy as np

class DATADetection(gdata.COCODetection):
    # TCT
    CLASSES = []

    # ury
    # CLASSES = ['eryth', 'leuko', 'epith', 'epithn', 'trichomonad', 'cluecell', 'spore', 'hypha']

    # voc
    # CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    #            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    #            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []
        im_aspect_ratios = []
        # lazy import pycocotools
        # try_import_pycocotools()
        from pycocotools.coco import COCO
        for split in self._splits:
            anno = os.path.join(self._root, self.annotation_dir, split) + '.json'
            _coco = COCO(anno)
            self._coco.append(_coco)
            classes = [c['name'] for c in _coco.loadCats(_coco.getCatIds())]
            if not classes == self.classes:
                raise ValueError("Incompatible category names with COCO: ")
            assert classes == self.classes
            json_id_to_contiguous = {
                v: k for k, v in enumerate(_coco.getCatIds())}
            if self.json_id_to_contiguous is None:
                self.json_id_to_contiguous = json_id_to_contiguous
                self.contiguous_id_to_json = {
                    v: k for k, v in self.json_id_to_contiguous.items()}
            else:
                assert self.json_id_to_contiguous == json_id_to_contiguous

            # iterate through the annotations
            image_ids = sorted(_coco.getImgIds())
            for entry in _coco.loadImgs(image_ids):
                abs_path = self._parse_image_path(entry)
                # print(abs_path)

                if not os.path.exists(abs_path):
                    raise IOError('Image: {} not exists.'.format(abs_path))
                label = self._check_load_bbox(_coco, entry)
                if not label:
                    continue
                im_aspect_ratios.append(float(entry['width']) / entry['height'])
                items.append(abs_path)
                labels.append(label)
        return items, labels, im_aspect_ratios

    def _parse_image_path(self, entry):
        """How to parse image dir and path from entry.

        Parameters
        ----------
        entry : dict
            COCO entry, e.g. including width, height, image path, etc..

        Returns
        -------
        abs_path : str
            Absolute path for corresponding image.

        """
        # abs_path = os.path.join(self._root, 'JPEGImages', entry['file_name'])
        abs_path = os.path.join(self._root, entry['file_name'])
        return abs_path
