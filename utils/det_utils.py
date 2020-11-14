import torch
import math


class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    :param batch_size_per_image: number of elements to be selected per image
    :param positive_fraction: percentage of positive elements per batch
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        """
        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        :param matched_idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.
        :return: pos_idx (list[tensor])
            neg_idx (list[tensor])
        """

        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            # positive sample if index >= 1
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            # negative sample if index == 0
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

            # number of positive samples
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples, used all positive samples
            num_pos = min(positive.numel(), num_pos)

            # number of negative samples
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples, used all negative samples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            # Returns a random permutation of integers from 0 to n - 1.
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


def encode_boxes(reference_boxes, proposals, weights):
    """
    Encode a set of proposals with respect to some reference boxes
    :param reference_boxes: reference boxes(gt)
    :param proposals: boxes to be encoded(anchors)
    :param weights:
    :return:
    """

    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    # Returns a new tensor with a dimension of size one inserted at the specified position.
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    # parse widths and heights
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1

    # center point
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    :param weights: 4-element tuple, represented calculation weights of x, y, h, w
    :param bbox_xform_clip: float, represented maximum of height and width
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        This class is inserted to calculate parameters of regression
        :param reference_boxes: gt bbox
        :param proposals: anchors bbox
        :return: regression parameters
        """

        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)

        # targets_dx, targets_dy, targets_dw, targets_dh
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some reference boxes
        :param reference_boxes: reference boxes
        :param proposals: boxes to be encoded
        :return:
        """

        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets

    def decode(self, rel_codes, boxes):
        """
        decode regression parameters
        :param rel_codes: bbox regression parameters
        :param boxes: anchors
        :return:
        """

        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)

        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)

        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        # map regression parameters into anchors to get coordinate
        pred_boxes = self.decode_single(
            rel_codes.reshape(box_sum, -1), concat_boxes
        )
        return pred_boxes.reshape(box_sum, -1, 4)

    def decode_single(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets, get the decoded boxes.
        :param rel_codes: encoded boxes (bbox regression parameters)
        :param boxes: reference boxes (anchors)
        :return:
        """
        boxes = boxes.to(rel_codes.dtype)

        # xmin, ymin, xmax, ymax
        widths = boxes[:, 2] - boxes[:, 0]   # anchor width
        heights = boxes[:, 3] - boxes[:, 1]  # anchor height
        ctr_x = boxes[:, 0] + 0.5 * widths   # anchor center x coordinate
        ctr_y = boxes[:, 1] + 0.5 * heights  # anchor center y coordinate

        wx, wy, ww, wh = self.weights  # default is 1
        dx = rel_codes[:, 0::4] / wx   # predicated anchors center x regression parameters
        dy = rel_codes[:, 1::4] / wy   # predicated anchors center y regression parameters
        dw = rel_codes[:, 2::4] / ww   # predicated anchors width regression parameters
        dh = rel_codes[:, 3::4] / wh   # predicated anchors height regression parameters

        # limit max value, prevent sending too large values into torch.exp()
        # self.bbox_xform_clip=math.log(1000. / 16)
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # xmin
        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymin
        pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        # xmax
        pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymax
        pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes


def set_low_quality_matches_(matches, all_matches, match_quality_matrix):
    """
    Produce additional matches for predictions that have only low-quality matches.
    Specifically, for each ground-truth find the set of predictions that have
    maximum overlap with it (including ties); for each prediction in that set, if
    it is unmatched, then match it to the ground-truth with which it has the highest
    quality value.
    """
    # For each gt, find the prediction with which it has highest quality
    highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)  # the dimension to reduce.

    # Find highest quality match available, even if it is low, including ties
    gt_pred_pairs_of_highest_quality = torch.nonzero(
        match_quality_matrix == highest_quality_foreach_gt[:, None]
    )
    # Example gt_pred_pairs_of_highest_quality:
    #   tensor([[    0, 39796],
    #           [    1, 32055],
    #           [    1, 32070],
    #           [    2, 39190],
    #           [    2, 40255],
    #           [    3, 40390],
    #           [    3, 41455],
    #           [    4, 45470],
    #           [    5, 45325],
    #           [    5, 46390]])
    # Each row is a (gt index, prediction index)
    # Note how gt items 1, 2, 3, and 5 each have two ties

    pre_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
    matches[pre_inds_to_update] = all_matches[pre_inds_to_update]


class Matcher(object):
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold  # 0.7
        self.low_threshold = low_threshold    # 0.3
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        calculate maximum iou between anchors and gt boxes, save index，
        iou < low_threshold: -1
        iou > high_threshold: 1
        low_threshold<=iou<high_threshold: -2
        :param match_quality_matrix:an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements
        :return:  matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """

        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)  # the dimension to reduce.
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD  # -1

        matches[between_thresholds] = self.BETWEEN_THRESHOLDS    # -2

        if self.allow_low_quality_matches:
            assert all_matches is not None
            set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches


def smooth_l1_loss(input, target, beta: float = 1. / 9, size_average: bool = True):
    """
    smooth_l1_loss for bbox regression
    :param input:
    :param target:
    :param beta:
    :param size_average:
    :return:
    """

    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()
