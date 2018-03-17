import torch
import torch.legacy.nn as nn
import torch.functional as F

def build_conv_block(dim, padding_type, use_instance_norm):
    conv_block = nn.Sequential()
    p = 0
    if padding_type == 'reflect':
        conv_block.add(nn.SpatialReflectionPadding(1,1,1,1))
    elif padding_type == 'replicate':
        conv_block.add(nn.SpatialReplicationPadding(1,1,1,1))
    elif padding_type == 'zero':
        p = 1

    conv_block.add(nn.SpationConvolution(dim, dim, 3, 3, 1, 1, p, p))

    if use_instance_norm == 1:
        conv_block.add(nn.InstanceNormalization(dim))
    else:
        conv_block.add(nn.SpatialBatchNormalization(dim))

    conv_block.add(F.ReLU(True))

    if padding_type == 'reflect':
        conv_block.add(nn.SpatialReflectionPadding(1, 1, 1, 1))
    elif padding_type == 'replicate':
        conv_block.add(nn.SpatialReplicationPadding(1, 1, 1, 1))

    conv_block.add(nn.SpatialConvolution(dim, dim, 3, 3, 1, 1, p, p))

    if use_instance_norm == 1:
        conv_block.add(nn.InstanceNormalization(dim))
    else:
        conv_block.add(nn.SpatialBatchNormalization(dim))

    return conv_block    


def build_model(opt):
    model = torch.nn.Sequential()
    prev_dim = 3
    arch = opt.arch.split(',')

    model = torch.nn.Sequential()

    arch_len = len(arch)

    for i in range(arch_len):
        v = arch[i]
        first_char = v[0]
        needs_relu = True
        needs_bn = True
        next_dim = None
        layer = None
        if first_char == 'c':
            f = int(v[1])
            p = (f - 1) / 2
            s = int(v[3])
            next_dim = int(v[5])
            if opt.padding_type == 'reflect':
                model.add(nn.SpatialReflectionPadding(p, p, p, p))
                p = 0
            elif opt.padding_type == 'replicate':
                model.add(nn.SpatialReplicationPadding(p, p, p, p))
                p = 0
            elif padding_type == 'none':
                p = 0
            layer = nn.SpatialConvolution(prev_dim, next_dim, f, f, s, s, p, p)
        elif first_char == 'f':
            f = int(v[1])
            p = (f - 1) / 2
            s = int(v[3])  
            a = s - 1
            next_dim = int(v[5])
            layer = nn.SpatialFullConvolution(prev_dim, next_dim, f, f, s, s, p, p, a, a)
        elif first_char == 'd':
            next_dim = int(v[1])
            layer = nn.SpatialConvolution(prev_dim, next_dim, 3, 3, 2, 2, 1, 1)
        elif first_char == 'U':
            next_dim = prev_dim
            scale = int(v[1])
            layer = nn.SpatialFullConvolution(prev_dim, next_dim, 3, 3, 2, 2, 1, 1, 1, 1)
            needs_bn = False
            needs_relu = True

        model.add(layer)
        if i == arch_len - 1:
            needs_bn = False
            needs_relu = False

        if needs_bn == True:
            if opt.use_instance_norm == 1:
                model.add(nn.InstanceNormalization(next_dim))
            else:
                model.add(nn.SpatialBatchNormalization(next_dim))
            
        if needs_relu == True:
            model.add(nn.ReLU(true))
        
        prev_dim = next_dim
    
    model.add(bb.Tanh())
    model.add(nn.MulConstant(opt.tanh_constant))
    model.add(nn.TotalVariation(opt.tv_strength))


