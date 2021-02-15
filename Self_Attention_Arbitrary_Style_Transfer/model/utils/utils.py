import torch
import io
import numpy as np
import os
from PIL import Image
import glob
#import pyheif  #not supported in windows
import torchvision
import torch
class ReadImages(object):
    '''
    ReadImages
     : Reads Images from path
     : PNG, JPEG, TIF, HEIC

    '''

    def __init__(self):
        pass

    def convertRGB(self, img_path):
        img = Image.open(img_path)
        assert(len(img.getbands()) == 4)
        img.load()  # required for png.split()
        rgb = Image.new("RGB", img.size, (255, 255, 255))
        rgb.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        return rgb, rgb.size

    def convertHEIC(self, img_path):
        for fn in glob.glob(img_path):
            with open(fn, 'rb') as f:
                d = f.read()
                heif_file = pyheif.read_heif(d)
                print(heif_file.mode)
                assert(heif_file is not None)
                assert(heif_file.mode in ['RGB', 'RGBA'])
                width, height = heif_file.size
                assert(width > 0)
                assert(height > 0)
                assert(len(heif_file.data) > 0)
                s = io.BytesIO()
                temp_img = Image.frombytes(
                    mode=heif_file.mode, size=heif_file.size, data=heif_file.data)
                if heif_file.mode == 'RGBA':
                    rgb.load()  # required for png.split()
                    rgb = Image.new("RGB", temp_img.size, (255, 255, 255))
                    # 3 is the alpha channel
                    rgb.paste(temp_img, mask=temp_img.split()[3])
                    return rgb, temp_img.size
    
                return temp_img, temp_img.size

    def checkFormat(self, img_path):
        im = Image.open(img_path)
        assert(im is not None)
        if img_path[-4:] == "HEIC":
            return self.convertHEIC(img_path)
        else:
            if len(im.getbands()) == 4:
                return self.convertRGB(img_path)
            else:
                assert(len(im.getbands()) == 3)
                return im, im.size

    def __call__(self, img_path):
        return self.checkFormat(img_path)


# Stores average loss
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Builds optimizer


def build_optimizer(model,
                    learning_rate,
                    optimizer_name='rmsprop',
                    weight_decay=1e-5,
                    epsilon=0.001,
                    momentum=0.9):
    """Build optimizer"""
    if optimizer_name == "sgd":
        # print("Using SGD optimizer.")
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=learning_rate,
                                    momentum=momentum,
                                    weight_decay=weight_decay)

    elif optimizer_name == 'rmsprop':
        # print("Using RMSProp optimizer.")
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=learning_rate,
                                        eps=epsilon,
                                        weight_decay=weight_decay,
                                        momentum=momentum
                                        )
    elif optimizer_name == 'adam':
        # print("Using Adam optimizer.")
        optimizer = torch.optim.Adam(model,
                                     lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def get_grid_image(X):
    X = X[:8]
    X = torchvision.utils.make_grid(X.detach().cpu(), nrow=X.shape[0]) * 0.5 + 0.5
    return X


def make_image(Xs, Xt, Y):
    Xs = get_grid_image(Xs)
    Xt = get_grid_image(Xt)
    Y = get_grid_image(Y)
    return torch.cat((Xs, Xt, Y), dim=1).numpy()

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def normal(feat, eps=1e-5):
    feat_mean, feat_std = calc_mean_std(feat, eps)
    normalized = (feat - feat_mean) / feat_std
    return normalized


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())

def chunkIt(seq, batch_size):
    '''
    This functions divides the list into batch.
    '''
    avg = batch_size
    out = []
    last = 0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def display_save(content, style, output,save_full=True, save_dir="./res", file_name="temp"):
    os.makedirs(save_dir, exist_ok=True)
    if save_full:
        pic = (torch.cat([content, style, output], dim=0))
    else:
        pic = (torch.cat([output], dim=0))
    grid = torchvision.utils.make_grid(pic, nrow=8, padding=2, pad_value=0,
                                       normalize=False, range=None, scale_each=False)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(
        1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save("{}/{}.jpg".format(save_dir, file_name))
