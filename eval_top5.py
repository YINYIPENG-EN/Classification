import argparse

import numpy as np
import torch
from PIL import Image

from detect import Classification, cvtColor, preprocess_input
from utils.utils import letterbox_image


class top5_Classification(Classification):
    def __init__(self, classes_path, model_name, input_shape, model_path, cuda):
        super(top5_Classification, self).__init__(classes_path=classes_path,
                                                  model_name=model_name,
                                                  input_shape=input_shape,
                                                  model_path=model_path,
                                                  cuda=cuda)
        self.input_shape = input_shape


    def detect_image(self, image):
        image       = cvtColor(image)
        image_data = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        image_data = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
            preds   = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        arg_pred = np.argsort(preds)[::-1] # argsort是从小到大排序，返回索引值，[::-1]从大到小
        arg_pred_top5 = arg_pred[:5]  # 取前5个
        return arg_pred_top5

def evaluteTop5(classfication, lines):
    correct = 0
    total = len(lines)
    for index, line in enumerate(lines):
        annotation_path = line.split(';')[1].split()[0]
        x = Image.open(annotation_path)
        y = int(line.split(';')[0])

        pred = classfication.detect_image(x)
        correct += y in pred
        if index % 100 == 0:
            print("[%d/%d]"%(index, total))
    return correct / total

if __name__ == '__main__':
    parse = argparse.ArgumentParser("eval top1")
    parse.add_argument('--cls_txt', type=str, default='model_data/cls_classes.txt')
    parse.add_argument('--model_name', type=str, default='mobilenet')
    parse.add_argument('--input_shape', type=list, default=[224,224])
    parse.add_argument('--weights', type=str, default='model_data/mobilenet_catvsdog.pth')
    parse.add_argument('--cuda', action='store_false')
    opt = parse.parse_args()
    print(opt)

    classes_path = opt.cls_txt
    model_name = opt.model_name
    input_shape = opt.input_shape
    model_path = opt.weights
    cuda = opt.cuda
    classfication = top5_Classification(classes_path, model_name, input_shape, model_path, cuda)
    with open("./cls_test.txt","r") as f:
        lines = f.readlines()
    top1 = evaluteTop5(classfication, lines)
    print("top-1 accuracy = %.2f%%" % (top1*100))
