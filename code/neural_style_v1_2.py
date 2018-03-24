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

def train(args):
    torch.manual_seed(args.seed)

    train_loader_LR, train_loader_HR, dataset_len = utils.make_dataset(args)

    transformer = utils.make_model(args)
    if args.updata:
        transformer.load_state_dict(torch.load(args.modeldir))
    vgg = utils.make_vggmodel(args)

    optimizer = Adam(transformer.parameters(), lr=args.lr)
    mse_loss = torch.nn.MSELoss()
    if args.cuda:
        mse_loss.cuda()

    for e in range(args.epochs):
        log_msg = "pix_weight = "+str(args.pix_weight)+"   content_weight = "+str(args.content_weight)
        print(log_msg)

        transformer.train()
        agg_content_loss = 0
        agg_pix_loss = 0
        count = 0

        for batch_id, ((x, x_),(style,y_)) in enumerate(zip(train_loader_LR,train_loader_HR)):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            pix_x_v = Variable(x)
            pix_s_v = Variable(style,requires_grad=False)

            pix_loss = 0
            content_loss = 0

            if args.cuda:
                x=x.cuda()
                style=style.cuda()
                pix_x_v=pix_x_v.cuda()
                pix_s_v=pix_s_v.cuda()

            output = transformer(pix_x_v)
            pix_loss = args.pix_weight * mse_loss(output, pix_s_v)

            vgg_s = Variable(style.clone(),requires_grad=False)

            vgg_x = utils.init_vgg_input(output)
            vgg_s = utils.init_vgg_input(vgg_s)


            feature_x = vgg(vgg_x)
            feature_s = vgg(vgg_s)

            f_s_v = Variable(feature_s[1].data, requires_grad=False)

            content_loss = args.content_weight * mse_loss(feature_x[1], f_s_v)

            total_loss = 0
            if args.pix_weight>0:
                total_loss+=pix_loss
            if args.content_weight>0:
                total_loss+=content_loss

            total_loss.backward()
            optimizer.step()
            agg_content_loss += content_loss.data[0]
            agg_pix_loss += pix_loss.data[0]
            up = 10000

            if(batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tpix: {:.6f}\ttotal: {:.6f}".format(\
                time.ctime(), e + 1, count, dataset_len,\
                agg_content_loss*up / (batch_id + 1),\
                agg_pix_loss *up/ (batch_id + 1),\
                (agg_content_loss + agg_pix_loss)*up / (batch_id + 1))
                print(mesg)         


    transformer.eval()
    transformer.cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
    args.content_weight) + "_SRCNN_" + str(args.srcnn) + ".model"
    save_model_path = "mymodel.model"
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)    
            



def stylize(args):
    content_image = utils.load_image_to_tensor(args.content_image,args.cuda)
    content_image.unsqueeze_(0)
    content_image = Variable(content_image)

    model = utils.make_model(args)
    model.load_state_dict(torch.load(args.model))

    output_image = model(content_image)
    output_image = output_image.data
    output_image.squeeze_(0)
    utils.save_tensor_to_image(output_image,args.output_image,args.cuda)

    

def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train",help="parser for training arguments")
    train_arg_parser.add_argument("--arch", type=str, default="-arch c9s1-64,R64,R64,R64,R64,u64,u64,c9s1-3", help="the archtechtrue of the network")
    train_arg_parser.add_argument("--epochs", type=int, default=2,help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,help="path to training dataset, the path should point to a folder "+"containing another folder with all the training images")
    train_arg_parser.add_argument("--upsample", type=int, default=4, required=True,help="the factor of the upsample 4 or 8")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--image-size", type=int, default=500,help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True, help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42, help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1.0,help="weight for content-loss, default is 1.0")
    train_arg_parser.add_argument("--pix-weight", type=float, default=1.0,help="weight for pix-loss, default is 1.0")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,help="learning rate, default is 0.001")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--srcnn", type=int, default=0,help="set it to 1 for running on srcnn")
    train_arg_parser.add_argument("--updata", type=int, default=0,help="set 1 to updata model")
    train_arg_parser.add_argument("--modeldir", type=str, default=0,help="set 1 to updata model")

    
    
    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--arch", type=str, default="-arch c9s1-64,R64,R64,R64,R64,u64,u64,c9s1-3", help="the archtechtrue of the network")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,help="saved model to be used for stylizing the image")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--srcnn", type=int, default=0,help="set it to 1 for running on srcnn")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        train(args)
    else:
        stylize(args)



if __name__ == "__main__":
    main()

