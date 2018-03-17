## fast-neural-style-super-resolution
implement fast-neural-style-super-resolution by pytorch

In this repository we only provide the source code.

## Setup
All code is implemented in [PyTorch](http://pytorch.org/).

See more details in PyTorch official website.

## Training a new model

Use `neural_style_vx_x.py` to train a new model.
Use `train` key word to set the script in train mode.

**Model options for training:**
- `--epoch`: How many epoch would you like to do. The default is 2.
- `--save-model-dir`: The directory that the model will be saved.
- `--cuda`: Set 1 to use GPU
- `--arch`: The architecture of the transform net. It isn't implemented yet.
- `--batch-size`: Default is 4
- `--dataset`: The dataset you used to train the model. The floder must contain another folder which contain all images.
- `--upsample`: The scale of the edge.
- `--image-size`: All images will be resize in the image_size
- `--seed`: Random seed for training
- `--content-weight`: Weight for content-loss, default is 1.0
- `--pix_weight`: Weight for pixel-loss, default is 1.0
- `--lr`: Learning rate. Default is 0.001
- `--log-interval`: Number of images after which the training loss is logged, default is 500
- `srcnn`: The architecture of the transform net. set 1 to use srcnn. It isn't implemented yet.


Use `eval` to super resolution a image.

**Modle options for super resolution**

- `--content-image`: The directory of the LR image.
- `--content-scale`: Factor for scaling down the content image.
- `--output-image`: Path for saving output image.
- `--model`: Saved model to be used for super resolution.
- `--cuda`: Set it to 1 for running on GPU, 0 for CPU