from utils.anchor_utils import generate_anchors


def generate_anchors_test():
    scales = [64, 128, 256]
    ratios = [0.5, 1.0, 2.0]
    generate_anchors(scales, ratios)


if __name__ == '__main__':
    generate_anchors_test()
