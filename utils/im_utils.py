import random
import torch
from torch.jit.annotations import List, Tuple
from torch import Tensor
from torchvision.transforms import functional as F


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return image, target



@torch.jit.script
class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        # type: (Tensor, List[Tuple[int, int]])
        """
        Arguments:
            tensors (tensor) padding后的图像数据
            image_sizes (list[tuple[int, int]])  padding前的图像尺寸
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        # type: (Device) # noqa
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)

