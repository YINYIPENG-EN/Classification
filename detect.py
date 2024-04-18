import argparse
from loguru import logger
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import matplotlib
import numpy as np
import torch
from torch import nn
from nets import get_model_from_name
from utils.utils import (cvtColor, get_classes,
                         preprocess_input, letterbox_image_opencv)
import cv2
matplotlib.use('TkAgg')  # 大小写无所谓 tkaGg ,TkAgg 都行

from tools.features_vision import CNN_Feature_Visualization

class Classification(object):
    def __init__(self, classes_path, model_name, input_shape, model_path, cuda):
        self.classes_path = classes_path
        self.backbone = model_name
        self.input_shape = input_shape
        self.model_path = model_path
        self.cuda = cuda
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.generate()

    def generate(self):
        if self.backbone != "vit":
            self.model  = get_model_from_name[self.backbone](num_classes = self.num_classes, pretrained = False)
        else:
            self.model  = get_model_from_name[self.backbone](input_shape = self.input_shape, num_classes = self.num_classes, pretrained = False)
        device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.model = self.model.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()

    def detect_image(self, image, visualize=False):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_data = letterbox_image_opencv(image, [self.input_shape[1], self.input_shape[0]])
        image_data = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo = torch.from_numpy(image_data)
            if self.cuda:
                photo = photo.cuda()
            preds = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy() # model(photo).shape = （batch_size,num_classes）
            if visualize:
                if isinstance(self.model, torch.nn.DataParallel):
                    self.model = self.model.module
                if self.backbone in ['mobilenet', 'vgg16']:
                    for name, m in self.model.features.named_children():
                        try:
                            photo = m(photo)
                            b, c, h, w = photo.shape
                            if h > 1 and w > 1:
                                CNN_Feature_Visualization(photo, name)
                        except:
                            continue
                else:  # resnet
                    for name, m in self.model.named_children():
                        try:
                            photo = m(photo)
                            b, c, h, w = photo.shape
                            if h > 1 and w > 1:
                                CNN_Feature_Visualization(photo, name)
                        except:
                            continue
        #   获得所属种类
        class_name = self.class_names[np.argmax(preds)]  # np.argmax(preds)获得preds中概率最大的索引
        probability = np.max(preds)  # 获得概率值
        return class_name, probability



if __name__ == '__main__':
    parse = argparse.ArgumentParser("classification detect")
    parse.add_argument('--weights', type=str, default='model_data/mobilenet_catvsdog.pth', help='weight path')
    parse.add_argument('--model_name', type=str, default='mobilenet', help='backbone name')
    parse.add_argument('--classes_path', type=str, default='model_data/cls_classes.txt')
    parse.add_argument('--cuda', action='store_false', default='use cuda')
    parse.add_argument('--input_shape', type=list, default=[224, 224], help='input shape')
    parse.add_argument('--img_path', type=str, default='img/cat.jpg', help='img path')
    parse.add_argument('--output', type=str, default='output', help='save path')
    parse.add_argument('--visualize', action='store_true', default=True, help='feature visualize')
    opt = parse.parse_args()
    logger.info(opt)
    if not os.path.exists(opt.output):
        os.makedirs(opt.output)
    if opt.visualize:
        os.makedirs('feat/', exist_ok=True)
    classification = Classification(opt.classes_path, opt.model_name, opt.input_shape, opt.weights, opt.cuda)
    img = cv2.imread(opt.img_path)
    res = classification.detect_image(img, opt.visualize)
    text = 'Class:%s Probability:%.2f%%' % (res[0], res[1]*100)
    axis = (10, 15)
    # font set
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (0, 255, 0)
    thickness = 1
    save_path = os.path.join(opt.output, f"{res[0]}_%.2f%%.jpg" % (res[1]*100))
    print(f"save img into {save_path}")
    cv2.putText(img, text, axis, font, fontScale, color, thickness)
    cv2.imshow("classification detect", img)
    cv2.waitKey(0)
    cv2.imwrite(save_path, img)
    cv2.destroyAllWindows()



