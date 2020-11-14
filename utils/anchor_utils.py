import torch
from torch import nn


def generate_anchors(scales, aspect_ratios, dtype=torch.float32, device="cpu"):
    """
     generate anchor template based on sizes and ratios, generated template is centered at [0, 0]
     :param scales: anchor sizes, in tuple[int]
     :param aspect_ratios: anchor ratios, in tuple[float]
     :param dtype: data type
     :param device: date device
     :return:
     """

    scales = torch.as_tensor(scales, dtype=dtype, device=device)
    aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
    h_ratios = torch.sqrt(aspect_ratios)
    w_ratios = 1.0 / h_ratios

    # [r1, r2, r3]' * [s1, s2, s3]
    # number of elements is len(ratios)*len(scales)
    ws = (w_ratios[:, None] * scales[None, :]).view(-1)
    hs = (h_ratios[:, None] * scales[None, :]).view(-1)

    # left-top, right-bottom coordinate relative to anchor center(0, 0)
    # anchor template is centered at [0, 0], shape [len(ratios)*len(scales), 4]
    base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

    return base_anchors.round()  # anchor will lose some precision here


class AnchorsGenerator(nn.Module):
    """
    anchor generator for feature maps according to anchor sizes and ratios
    :param sizes: anchor sizes, in tuple[int]
    :param aspect_ratios: anchor ratios, in tuple[float]
    :return:
    """

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorsGenerator, self).__init__()

        # assert len(sizes) == len(aspect_ratios), 'anchor sizes must equal to anchor ratios!'

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def set_cell_anchors(self, dtype, device):
        """
        generate template template
        :param dtype: data type
        :param device: data device
        :return:
        """
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None

        # generate anchor template
        cell_anchors = [generate_anchors(sizes, aspect_ratios, dtype, device)
                        for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        # calculate the number of anchors per feature map, for k in origin paper
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def grid_anchors(self, feature_map_sizes, strides):
        """
        compute anchor coordinate list in origin image, mapped from feature map
        :param feature_map_sizes: feature map sizes
        :param strides: strides between origin image and anchor
        :return:
        """

        anchors = []
        cell_anchors = self.cell_anchors  # anchor template
        assert cell_anchors is not None

        # for every resolution feature map, like fpn
        for size, stride, base_anchors in zip(feature_map_sizes, strides, cell_anchors):
            f_p_height, f_p_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center...]
            # x_center in origin image
            shifts_x = torch.arange(0, f_p_width, dtype=torch.float32, device=device) * stride_width

            # y_center in origin image
            shifts_y = torch.arange(0, f_p_height, dtype=torch.float32, device=device) * stride_height

            # torch.meshgrid will output grid
            # shape: [grid_height, grid_width]
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors.append(shifts_anchor.reshape(-1, 4))

        return anchors  # List[Tensor(all_num_anchors, 4)]

    def cached_grid_anchors(self, feature_map_size, strides):
        """
        cached all anchor information
        :param feature_map_size: feature map size after backbone feature extractor
        :param strides: strides between origin image size and feature map size
        :return:
        """

        key = str(feature_map_size) + str(strides)
        # self._cache is a dictionary type
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(feature_map_size, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list, feature_maps):
        """
        get feature map sizes
        :param image_list:
        :param feature_maps:
        :return:
        """

        feature_map_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])

        # get input image sizes
        image_size = image_list.tensors.shape[-2:]

        # get dtype and device
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        # compute map stride between feature_maps and input images
        strides = [[torch.tensor(image_size[0] / g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] / g[1], dtype=torch.int64, device=device)] for g in feature_map_sizes]

        # get anchors template according size and aspect_ratios
        self.set_cell_anchors(dtype, device)

        # get anchor coordinate list in origin image, according to map
        anchors_over_all_feature_maps = self.cached_grid_anchors(feature_map_sizes, strides)

        anchors = []
        # for every image and feature map in a batch
        for i, (_, _) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            # for every resolution feature map like fpn
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)

        # concat every resolution anchors, like fpn
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]

        self._cache.clear()
        return anchors
