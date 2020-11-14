import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from config.test_config import test_cfg
from dataloader.coco_dataset import coco
from utils.draw_box_utils import draw_box
from utils.train_utils import create_model


def test():
    model = create_model(num_classes=test_cfg.num_classes)

    model.cuda()
    weights = test_cfg.model_weights

    checkpoint = torch.load(weights, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # read class_indict
    data_transform = transforms.Compose([transforms.ToTensor()])
    test_data_set = coco(test_cfg.data_root_dir, 'test', '2017', data_transform)
    category_index = test_data_set.class_to_coco_cat_id

    index_category = dict(zip(category_index.values(), category_index.keys()))

    original_img = Image.open(test_cfg.image_path)
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        predictions = model(img.cuda())[0]
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("No target detected!")

        draw_box(original_img,
                 predict_boxes,
                 predict_classes,
                 predict_scores,
                 index_category,
                 thresh=0.3,
                 line_thickness=3)
        plt.imshow(original_img)
        plt.show()


if __name__ == "__main__":
    version = torch.version.__version__[:5]
    print('torch version is {}'.format(version))
    os.environ["CUDA_VISIBLE_DEVICES"] = test_cfg.gpu_id
    test()
