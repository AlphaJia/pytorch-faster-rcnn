import math

import torch
from torch import nn
from torch.jit.annotations import List, Tuple

from utils.im_utils import ImageList


def torch_choice(l):
    index = int(torch.empty(1).uniform_(0., float(len(l))).item())
    return l[index]


def max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def batch_images(images, size_divisible=32):
    """
    batched images
    :param images: a set of images
    :param size_divisible: ratio of height/width to be adjusted
    :return: batched tensor image
    """

    max_size = max_by_axis([list(img.shape) for img in images])

    stride = float(size_divisible)

    max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
    max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

    # [batch, channel, height, width]
    batch_shape = [len(images)] + max_size

    batched_imgs = images[0].new_full(batch_shape, 0)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    return batched_imgs


class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN model.
    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    :param min_size: minimum size of input image
    :param max_size: maximum size of input image
    :param image_mean: image mean
    :param image_std: image std
    """

    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        """
        resize input image to specified size and transform for target
        :param image: input image
        :param target: target related info, like bbox
        :return:
            image: resized image
            target: resized target
        """

        # image shape is [channel, height, width]
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = float(torch.min(im_shape))
        max_size = float(torch.max(im_shape))
        if self.training:
            size = float(torch_choice(self.min_size))
        else:
            size = float(self.min_size[-1])
        scale_factor = size / min_size

        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size

        image = torch.nn.functional.interpolate(
            image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        return image, target

    def postprocess(self, result, image_shapes, original_image_sizes):
        """
        post process of predictions, mainly map bboxed coordinates to original image
        :param result: predictions result
        :param image_shapes: image size after preprocess
        :param original_image_sizes: original image size
        :return:
        """

        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
        return result

    def forward(self, images, targets=None):
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image = self.normalize(image)
            image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        # save resized image size
        image_sizes = [img.shape[-2:] for img in images]
        images = batch_images(images)
        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])

        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets


def resize_boxes(boxes, original_size, new_size):
    """
    resize bbox to original image based on stride
    :param boxes: predicted bboxes
    :param original_size: original image size
    :param new_size: rescaled image size
    :return:
    """
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratios_height, ratios_width = ratios

    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)








