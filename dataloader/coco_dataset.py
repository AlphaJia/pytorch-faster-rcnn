from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from pycocotools.coco import COCO


class coco(Dataset):
    def __init__(self, root_dir, image_set, year, transforms=None):

        self._root_dir = root_dir
        self._year = year
        self._image_set = image_set
        self._data_name = image_set + year
        self._json_path = self._get_ann_file()
        self._transforms = transforms

        # load COCO API
        self._COCO = COCO(self._json_path)

        with open(self._json_path) as anno_file:
            self.anno = json.load(anno_file)

    def __len__(self):
        return len(self.anno)

    def _get_ann_file(self):
        prefix = 'instances' if self._image_set.find('test') == -1 else 'image_info'
        return os.path.join(self._root_dir, 'annotations', prefix + '_' + self._image_set + self._year + '.json')

    def _image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   images/train2014/COCO_train2014_000000119993.jpg
        file_name = (str(index).zfill(12) + '.jpg')
        image_path = os.path.join(self._root_dir, self._data_name, file_name)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def __getitem__(self, idx):
        a = self.anno['images'][idx]
        image_idx = a['id']
        img_path = os.path.join(self._root_dir, self._data_name, self._image_path_from_index(image_idx))
        image = Image.open(img_path)

        width = a['width']
        height = a['height']

        annIds = self._COCO.getAnnIds(imgIds=image_idx, iscrowd=None)
        objs = self._COCO.loadAnns(annIds)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)

        iscrowd = []
        for ix, obj in enumerate(objs):
            cls = obj['category_id']
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            iscrowd.append(int(obj["iscrowd"]))

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        image_id = torch.tensor([image_idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = gt_classes
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


if __name__ == '__main__':
    a = coco("/data/jiapf/code/pytorch-cpn/data/COCO2017/",
             "train",
             '2017')
    a.__getitem__(9)
