import os

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.serialization import load_lua
from torchvision import transforms

from vgg16 import Vgg16
from Image_utils import AddMyGauss

from torch.utils.data import DataLoader
from torchvision import datasets
import transformer_net as tn
from SRCNN import SRCNN

import argparse
import os
import sys
import time

from scipy.ndimage import filters

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils import model_zoo
from normalhead import LossNetwork
import Image_utils as iu
from SRCNN import SRCNN


import utils
from transformer_net import TransformerNet
import torchvision.models.vgg as vggNetWork
import torchvision

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


def init_vgg_input(batch):
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std = tensortype(batch.data.size())
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch = batch.sub(Variable(mean))
    batch = batch.div(Variable(std))
    return batch


def make_dataset(args):
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        kwargs1 = {'num_workers':0, 'pin_memory':False}
        kwargs2 = {'num_workers':0, 'pin_memory':False}
    else:
        kwargs1 = {}
        kwargs2 = {}

    transform_LR = None
    transform_HR = None

    transform_HR = transforms.Compose([ transforms.FiveCrop((args.size,args.size),\
                                        transforms.ToTensor()\
                                       ])
    transform_LR = transforms.Compose([ transforms.FiveCrop((args.size,args.size)),\
                                        AddMyGauss(),\
                                        transforms.ToTensor()\
                                        ])
    """
    transform_HR = transforms.Compose([\
                                        transforms.Resize((args.image_size,args.image_size),interpolation=3),\
                                        #transforms.CenterCrop((int(args.image_size/2),int(args.image_size/2))),\
                                        transforms.ToTensor()
                                       ])
    if args.srcnn:
        transform_LR = transforms.Compose([\
                                        transforms.Resize((args.image_size,args.image_size),interpolation=3),\
                                        transforms.CenterCrop((int(args.image_size/2),int(args.image_size/2))),\
                                        AddMyGauss(),\
                                        transforms.ToTensor()
                                       ])
    else:
        transform_LR = transforms.Compose([\
                                        transforms.Resize((int(args.image_size/args.upsample),int(args.image_size/args.upsample)),interpolation=3),\
                                        transforms.Resize((int(args.image_size),int(args.image_size)),interpolation=3),\
                                        #transforms.CenterCrop((int(args.image_size/args.upsample/2),int(args.image_size/args.upsample/2))),\
                                        AddMyGauss(),\
                                        transforms.ToTensor()
                                       ])
    """

    train_dataset_HR = datasets.ImageFolder(args.dataset, transform_HR)
    train_loader_HR = DataLoader(train_dataset_HR, batch_size = args.batch_size, **kwargs1)

    train_dataset_LR = datasets.ImageFolder(args.dataset, transform_LR)
    train_loader_LR = DataLoader(train_dataset_LR, batch_size = args.batch_size, **kwargs2)

    return train_loader_LR, train_loader_HR,len(train_dataset_HR)


def make_model(args):
    transformer = None
    if args.srcnn:
        transformer = SRCNN()
    else:
        transformer = tn.TransformerNet(args.arch)
    if args.cuda:
        transformer.cuda()
    return transformer


def make_vggmodel(args):
    vggmodel = torchvision.models.vgg.vgg16(pretrained=True)
    if args.cuda:
        vggmodel=vggmodel.cuda()
    vgg = LossNetwork(vggmodel)
    vgg.eval()
    del vggmodel
    if args.cuda:
        vgg.cuda()
    return vgg
