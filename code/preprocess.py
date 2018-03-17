import torch

class PREPROCESS:

    def check_input(img):
        if img.dim() != 4:
            raise NameError("img must be N x C x H x W")
        if img.size(1) != 3:
            raise NameError('img must have three channels')

    resnet = []
    resnet_mean = [0.484, 0.456, 0.406]
    renet_std = [0.229, 0.224, 0.225]

    def resnet_preprocess(img):
        check_input(img)
        mean = img.new(resnet_mean).view(1, 3, 1, 1).expand(img)
        std = img.new(resnet_std).view(1, 3, 1, 1).expand(img)
        return (img - mean) / std

    def resnet_deprocess(img):
        check_input(img)
        mean = img.new(resnet_mean).view(1, 3, 1, 1).expand(img)
        std = img.new(resnet_std).view(1, 3, 1, 1).expand(img)
        return img * std + mean

    vgg_mean = [103.939, 116.779, 123.68]

    
    def vgg_proprocess(img):
        check_input(img)
        mean = img.new(vgg_mean).view(1, 3, 1, 1).expand(img)
        prem = torch.LongTensor()

    #def vgg_proprocess(img):
    #    check_input(imp)
    #    mean = img.new(vgg_mean).view(1, 3, 1, 1).expand(img)
    #    perm = torch.LongTensor(3, 2, 1)
    #    return img[]