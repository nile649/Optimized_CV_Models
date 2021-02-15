import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageOps
import collections
import random
from scipy import ndimage
import pdb


class Transform(object):
    """
PolyGAN DataTransform
    """
    def __init__(self):
        pass
    def test_transform(self,size):
        transform_list = []
        if size!=None:
            transform_list.append(transforms.Resize((size,size)))
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
        return transform
    
    def __call__(self,size):
        return self.test_transform(size)
        
