import torch
from time import sleep
from model.utils import utils,feather
from model import layer, Vgg19
import pdb
class Inference(object):
    def __init__(self,args,**kwargs):
        super(Inference, self).__init__()
        self.args = args
        pdb.set_trace()
        self.vgg = Vgg19(self.args.path_vgg,self.args.device)
        self.decoder = layer.decoder().to(self.args.device)
        self.sa_module = layer.Self_Attention_Module(512).to(self.args.device)
        self.decoder.load_state_dict(torch.load(self.args.decoder_path))
        self.sa_module.load_state_dict(torch.load(self.args.transform_path))
        self.feather = feather.Feather_()

    def __featherParameters__(self,cimg,width,height):

        basic_width = 512
        padding_width = 32
        self.feather.set_IMG_Attr(basic_width = 512,\
        padding_width = 32,\
        block_width = basic_width+padding_width*2,\
        blending_width = padding_width,\
        width = cimg.shape[3],\
        height = cimg.shape[2],\
        width_expand = cimg.shape[3]+2*padding_width,\
        height_expand = cimg.shape[2]+2*padding_width,\
        og_width = width,\
        og_height = height,\
        device = self.args.device)  


        
    def __StyleTransfer__(self, content, style, alpha):
        assert (0.0 <= alpha <= 1.0)
        self.content_f, self.style_f = self.vgg(content, style)
        Fccc = self.sa_module(self.content_f, self.content_f)
        feat = self.sa_module(self.content_f, self.style_f)
        feat = feat * alpha + Fccc * (1 - alpha)
        return self.decoder(feat)


    def __call__(self,content=None, style=None, alpha=1.0,width=512,height=512,min_batch=1):
        self.__featherParameters__(content,width,height)
        style = style.repeat(int(min_batch),1,1,1)
        imageList = self.feather.breakImg(content)
        subimage_list = utils.chunkIt(imageList,int(min_batch))
        out_list = []
        for tensorList in subimage_list:
            tensor_ct = torch.stack(tensorList,1)
            tensor_ct = tensor_ct[0,:,:,:,:]
            with torch.no_grad():
                temp = self.__StyleTransfer__(tensor_ct,style,alpha)
            image_output = temp
            [out_list.append(x.unsqueeze(0)) for x in image_output]
        image = self.feather.attachImg(out_list)
        return image
