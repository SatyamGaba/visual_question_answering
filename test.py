import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data_loader import get_loader
from models import VqaModel, SANModel
#from resize_images import resize_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    image = args.image_path.to(device)
    question = args.question.to(device)
    image = cv2.resize(image, size=224, interpolation = cv2.INTER_AREA)
    #resize_image(image, size = 224)
    model = torch.load(args.saved_model)
    #torch.cuda.empty_cache()
    model.eval()
    output = model(image, question)
    _, pred_exp1 = torch.max(output, 1)
    print(pred_exp1)


if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type = str, required=True)
    parser.add_argument('--question', type = str, required=True)
    parser.add_argument('--saved_model', type = str, required=True)
       
    args = parser.parse_args()
    main(args)
