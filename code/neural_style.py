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
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        kwargs1 = {'num_workers':0, 'pin_memory':False}
        kwargs2 = {'num_workers':0, 'pin_memory':False}
    else:
        kwargs1 = {}
        kwargs2 = {}

    """ 
    ### for SR the transdataset should be be HR and LR. 
    transform = transforms.Compose([transforms.Scale(args.image_size), transforms.CenterCrop(args.image_size), transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, **kwargs)
    """
    
   
  
    transform_LR = None
    transform_HR = None
    if args.srcnn:
        transform_HR = transforms.Compose([transforms.Resize((args.image_size, args.image_size), interpolation=3),transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])
        transform_LR = transforms.Compose([iu.AddMyGauss(), transforms.Resize((int(args.image_size),int(args.image_size)),interpolation=3),transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])
    else:
        transform_HR = transforms.Compose([transforms.CenterCrop((args.image_size,args.image_size)),transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])
        transform_LR = transforms.Compose([iu.AddMyGauss(),transforms.CenterCrop((args.image_size,args.image_size)),transforms.Resize((int(args.image_size/args.upsample),int(args.image_size/args.upsample)),interpolation=3), transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])
    #transform_LR = transforms.Compose([iu.AddMyGauss(),transforms.Resize((int(args.image_size/args.upsample),int(args.image_size/args.upsample)),interpolation=3), transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])
    
    
    #transform_HR = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])
    train_dataset_HR = datasets.ImageFolder(args.dataset, transform_HR)
    train_loader_HR = DataLoader(train_dataset_HR, batch_size = args.batch_size, **kwargs1)
    #print(args.image_size/args.upsample)
    train_dataset_LR = datasets.ImageFolder(args.dataset, transform_LR)
    train_loader_LR = DataLoader(train_dataset_LR, batch_size = args.batch_size, **kwargs2)
    
    transformer = None
    if args.srcnn:
        transformer = SRCNN()
    else:
        transformer = TransformerNet(args.arch)
    #transformer = TransformerNet()
    optimizer = Adam(transformer.parameters(), lr=args.lr)
    mse_loss = torch.nn.MSELoss()

    """
    vggmodel = torchvision.models.vgg.vgg16(pretrained=True)
    if args.cuda:
        vggmodel=vggmodel.cuda()
    vgg = LossNetwork(vggmodel)
    vgg.eval()
    del vggmodel
    """
    if args.cuda:
        transformer.cuda()
        #vgg.cuda()
    
    #style = utils.tensor_load_rgbimage(args.style_image, size = args.style_size)
    #style = style.repeat(args.batch_size, 1, 1, 1)
    #style = utils.preprocess_batch(style)
    

    #if args.cuda:
    #    style=style.cuda()
    #style_v = utils.subtract_imagenet_mean_batch(Variable(style, volatile = True))
    #if args.cuda:
    #    style_v=style_v.cuda()
    #features_style = vgg(style_v)
    #gram_style = [utils.gram_matrix(y) for y in features_style]
    #log_msg = "pix_weight = "+args.pix_weight+"   content_weight = "+args.content_weight
    
    
    for e in range(args.epochs):
        log_msg = "pix_weight = "+str(args.pix_weight)+"   content_weight = "+str(args.content_weight)
        print(log_msg)
        transformer.train()
        agg_content_loss = 0
        agg_style_loss = 0
        count = 0
        for batch_id, ((x, x_),(style,y_)) in enumerate(zip(train_loader_LR,train_loader_HR)):
            #(y,y_) = train_loader_HR[batch_id]
            
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            
            #style = utils.tensor_load_rgbimage(args.style_image, size=args.style_size)
            #style = style.repeat(args.batch_size, 1, 1, 1)
            #gram_style = [utils.gram_matrix(y) for y in features_style]

            x = utils.preprocess_batch(x)
            style = utils.preprocess_batch(style)
            
            x = Variable(x)

            if args.cuda:
                x = x.cuda()

            y = transformer(x)

            yy = Variable(style)

            if args.cuda:
                yy = yy.cuda()

            pix_loss = args.pix_weight * mse_loss(y,yy)

            """
            xc = Variable(style.clone())
            if(args.cuda):
                xc = xc.cuda()
            y = utils.subtract_imagenet_mean_batch(y)
            xc = utils.subtract_imagenet_mean_batch(xc)

            features_y = vgg(y)
            features_xc = vgg(xc)

            f_xc_c = Variable(features_xc[1].data)

            content_loss = args.content_weight * mse_loss(features_y[1], f_xc_c)
            """ 
            #style_loss = 0;

            """
            for m in range(len(features_y)):
                gram_s = Variable(gram_style[m].data, requires_grad = False)
                gram_y = utils.gram_matrix(features_y[m])
                style_loss += args.style_weight * mse_loss(gram_y, gram_s[:n_batch, :, :])
            """

           
            total_loss = pix_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss +=  pix_loss.data[0]
            #agg_style_loss += style_loss.data[0]

            if(batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(\
                time.ctime(), e + 1, count, len(train_dataset_LR),\
                agg_content_loss / (batch_id + 1),\
                agg_style_loss / (batch_id + 1),\
                (agg_content_loss + agg_style_loss) / (batch_id + 1))
                print(mesg)
    
    transformer.eval()
    transformer.cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
    args.content_weight) + "_SRCNN_" + str(args.srcnn) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
    try:
        if not os.path.exists(args.vgg_model_dir):
            os.makedirs(args.vgg_model_dir)
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

def stylize(args):
    #content_image = utils.tensor_load_rgbimage(args.content_image, scale = args.content_scale)
    #content_image = content_image.unsqueeze(0)
    content_image = None
    if args.srcnn:
        content_image = utils.tensor_load_rgbimage(args.content_image, scale=args.upsample)
    else:
        content_image = utils.tensor_load_rgbimage(args.content_image)
    content_image.unsqueeze_(0)
    if args.cuda:
        content_image = content_image.cuda()
    content_image = Variable(utils.preprocess_batch(content_image), volatile=True)

    style_model = None
    if args.srcnn:
        style_model = SRCNN()
    else:
        style_model = TransformerNet(args.arch)
    ##style_model = TransformerNet()
    style_model.load_state_dict(torch.load(args.model))

    if args.cuda:
        style_model.cuda()
    
    output = style_model(content_image)
    utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)

def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

   
    train_arg_parser = subparsers.add_parser("train",
                                             help="parser for training arguments")
    train_arg_parser.add_argument("--arch", type=str, default="c9s1-64,R64,R64,R64,R64,u64,u64,c9s1-3", help="the archtechtrue of the network")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--upsample", type=int, default=4, required=True,
                                  help="the factor of the upsample 4 or 8")
    # SR dont need this option
    #train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
    #                              help="path to style-image") this vgg model is from LUA. we ues vgg from pytoch with pretrain
    #train_arg_parser.add_argument("--vgg-model-dir", type=str, required=True,
    #                              help="directory for vgg, if model is not present in the directory it is downloaded")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--srcnn", type=int, default=0
                                  ,help="1 use srcnn")
    train_arg_parser.add_argument("--image-size", type=int, default=288,
                                  help="size of training images, default is 288 X 288")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True, help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42, help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1.0,
                                  help="weight for content-loss, default is 1.0")
    train_arg_parser.add_argument("--pix-weight", type=float, default=0.0,
                                  help="weight for pix-loss, default is 0.0")
    #SR dont need this option
    #train_arg_parser.add_argument("--style-weight", type=float, default=5.0,
    #                              help="weight for style-loss, default is 5.0")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 0.001")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--srcnn", type=int, default=0,
                                 help="1 use srcnn")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--arch", type=str, default="-arch c9s1-64,R64,R64,R64,R64,u64,u64,c9s1-3", help="the archtechtrue of the network")
    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        #check_paths(args)
        train(args)
    else:
        stylize(args)






if __name__ == "__main__":
    main()
