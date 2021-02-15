
import argparse
import os
import timeit
from data.data import dataiter
from model import Inference
from model import utils
from PIL import Image
import numpy as np
import platform

ostype = platform.platform()[:7]
class options():
    def __init__(self):
        self.decoder_path = "./model/pre_trained_model/decoder_iter_100000.pth"
        self.transform_path = "./model/pre_trained_model/sa_module_iter_100000.pth"
        self.path_vgg = "./model/pre_trained_model/vgg_normalised.pth"
        self.lr = 0.0001
        self.optimizer = "adam"
        self.content_weight = 1
        self.style_Weight = 10
        self.mode = "Test"
        self.crop = False
        self.num_workers = 4
        self.batch_size = 1
        self.device = "cuda"
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str,default='./testimages/normal.jpg') # number of workers/thread to use for loading dat
    parser.add_argument('--style', type=str,default = './static/TheScream.jpg') # batch size
    parser.add_argument("--alpha", type=float,default = 0.75)
    parser.add_argument("--save_full", type=bool,default = False)
    parser.add_argument("--res",type=str, default = './')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = options()
    arg = get_opt()
    inference = Inference(opt)
    image = Image.open(arg.content).convert("RGB") 
    height_ = image.size[0]
    width_ = image.size[1]

    size_list = [512,1024,2048,4096]
    di = {512:1,1024:4,2048:4,4096:4}
    max_ = np.max((int(height_),int(width_)))
    sz = min(size_list, key=lambda x:abs(x-max_))
    min_batch = di[sz]
    file_name = arg.content.split("/")[-1][:-4]+ "_" + arg.style.split("/")[-1][:-4] + "_" + "{}".format(arg.alpha)
    
    start = timeit.default_timer()
    content = dataiter(opt,sz,arg.content)
    cimg = next(content)
    style = dataiter(opt,512,arg.style)
    simg = next(style)
    if ostype=="Windows":
        res = inference(cimg.to(opt.device),simg.to(opt.device),arg.alpha,height_,width_,min_batch)
    else:
        res = inference(cimg,simg,1,height_,width_,min_batch)
    utils.display_save(cimg,simg,res,False,'./',file_name)
    end = timeit.default_timer()

    print("Finish : {} seconds".format(end-start))
#print("time taken {}".)
                                                
                                                                  
                                                                  
