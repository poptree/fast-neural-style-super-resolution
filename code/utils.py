import os

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.serialization import load_lua
from torchvision import transforms

from vgg16 import Vgg16

vgg_mean = (0.485, 0.456, 0.406)
vgg_std = (0.229, 0.224, 0.225)


def transforms_compose_norm(size=None):
    seq = []
    if size!=None:
        seq.append(transforms.Resize(size,interpolation=3))
    seq.append(transforms.ToTensor())
    #seq.append(transforms.Normalize(mean=vgg_mead,std=vgg_std))
    return transforms.Compose(seq)


def load_image_to_tensor(filename,cuda, size=None):
    img = Image.open(filename)
    com = transforms_compose_norm(size)
    img = com(img)
    if cuda:
        img=img.cuda()
    return img


def save_tensor_to_image(tensor,filename,cuda):
    if cuda:
        tensor = tensor.cpu()
    img = transforms.ToPILImage()(tensor)
    img.save(filename)


def init_vgg_input(tensor):
    norm = transforms.Normalize(mean=vgg_mean,std=vgg_std)
    epoch = tensor.size()[0]
    for i in range(epoch):
        norm(tensor[i])
    return tensor