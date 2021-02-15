from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from random import randrange, seed
import os
import pdb
import copy

class Encrypt:
    def __init__(self,utils,EncryptMethod=None):
        self.utils = utils()
        self.object = EncryptMethod(utils)
        
    def __call__(self,im,passwords="yOs0hfhygk",encrypt=1):
        size=im.size
        columns=size[0]
        rows=size[1]
        return self.object(im,columns,rows,passwords,encrypt)

    