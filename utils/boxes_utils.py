import torch


def nms(boxes, scores, iou_threshold):
    """
     Performs non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an IoU greater than iou_threshold with another (higher scoring)
    box.
    :param boxes: Tensor[N, 4]), boxes to perform NMS on. They are expected to be in (x1, y1, x2, y2) format
    :param scores: Tensor[N], scores for each one of the boxes
    :param iou_threshold: float, discards all overlapping boxes with IoU < iou_threshold
    :return: int64 tensor with the indices of the elements that have been kept by NMS, sorted in decreasing order of scores
    """

    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.
    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories
    :param boxes: Tensor[N, 4], boxes where NMS will be performed. They are expected to be in (x1, y1, x2, y2) format
    :param scores:  Tensor[N], scores for each one of the boxes
    :param idxs: Tensor[N], indices of the categories for each one of the boxes.
    :param iou_threshold: float, discards all overlapping boxes, with IoU < iou_threshold
    :return: int64 tensor with the indices of the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()

    # to(): Performs Tensor dtype and/or device conversion
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def remove_small_boxes(boxes, min_size):
    """
    Remove boxes which contains at least one side smaller than min_size.
    :param boxes: boxes in (x1, y1, x2, y2) format
    :param min_size: minimum size
    :return: indices of the boxes that have both sides
            larger than min_size
    """

    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    # nonzero(): Returns a tensor containing the indices of all non-zero elements of input
    keep = keep.nonzero().squeeze(1)
    return keep


def clip_boxes_to_image(boxes, size):
    """
    Clip boxes so that they lie inside an image of size `size`.
    :param boxes: boxes in (x1, y1, x2, y2) format
    :param size: size of the image
    :return: clipped_boxes (Tensor[N, 4])
    """

    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]  # x1, x2
    boxes_y = boxes[..., 1::2]  # y1, y2
    height, width = size

    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    :param boxes:  boxes for which the area will be computed. They
                   are expected to be in (x1, y1, x2, y2) format
    :return: area for each box
    """

    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """
     Calculate intersection-over-union (Jaccard index) of boxes.
     Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    :param boxes1: boxes1 (Tensor[N, 4])
    :param boxes2: boxes2 (Tensor[M, 4])
    :return: iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def permute_and_flatten(layer, N, A, C, H, W):
    """
    adjust tensor order，and reshape
    :param layer: classification or bboxes parameters
    :param N: batch_size
    :param A: anchors_num_per_position
    :param C: classes_num or bbox coordinate
    :param H: height
    :param W: width
    :return: Tensor after adjusting order and reshaping
    """

    # [batch_size, anchors_num_per_position * (C or 4), height, width]
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)  # [N, H, W, -1, C]
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    """
    Adjust box classification and bbox regression parameters order and reshape
    :param box_cls: target prediction score
    :param box_regression: bbox regression parameters
    :return: [N, -1, C]
    """

    box_cls_flattened = []
    box_regression_flattened = []

    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        # [batch_size, anchors_num_per_position * classes_num, height, width], class_num is equal 2
        N, AxC, H, W = box_cls_per_level.shape
        # [batch_size, anchors_num_per_position * 4, height, width]
        Ax4 = box_regression_per_level.shape[1]
        # anchors_num_per_position
        A = Ax4 // 4
        # classes_num
        C = AxC // A

        # [N, -1, C]
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        # [N, -1, C]
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)  # start_dim, end_dim
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression

