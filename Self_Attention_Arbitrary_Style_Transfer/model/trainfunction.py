import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision
from torch.utils.tensorboard import SummaryWriter
from time import sleep
from utils import utils
from loss import Loss
from layers import layers,Vgg19

class Train(StyleTransfer):
    def __init__(self,args,**kwargs):
        super(Inference, self).__init__()
        self.args = args
        self.init(**kwargs)
    
    def init(**kwargs):
        for arg in kwargs.items():
            if hasattr(self.args, arg[0]):
                setattr(self.args,arg[0],arg[1])
        self.vgg = Vgg19(self.args.path_vgg,self.args.device)
        self.decoder = layers.decoder()
        self.sa_module = layers.Self_Attention_Module(512)
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.sa_module.load_state_dict(torch.load(transform_path))
        self.criterion = Loss()
        self.optimizer = utils.build_optimizer(model=[
            {'params': self.decoder.parameters()},
            {'params': self.sa_module.parameters()}],
            learning_rate=self.opt.lr,
            optimizer_name=self.opt.optimizer
        )
        
    def __forward__(self):
        self.optimizer.zero_grad()
        style_feats = self.vgg.encode_with_intermediate(style)
        content_feats = self.vgg.encode_with_intermediate(content)
        Ics = self.decoder(self.sa_module(content_feats, style_feats))
        Ics_feats = self.vgg.encode_with_intermediate(Ics)
        
        Icc = self.decoder(self.sa_module(content_feats, content_feats))
        Iss = self.decoder(self.sa_module(style_feats, style_feats)) 
        
        Icc_feats=self.vgg.encode_with_intermediate(Icc)
        Iss_feats=self.vgg.encode_with_intermediate(Iss)    
        loss_c, loss_s, l_identity1, l_identity2 = self.criterion()
        return content,style,\
            style_feats,content_feats,\
            Ics, Ics_feats, Icc, Icc_feats, Iss, Iss_feats
    def __loss__(self):
        loss_c, loss_s, l_identity1, l_identity2 = self.criterion(
                    self.__forward__(self.content, self.style))

        loss_c = self.args.content_weight * loss_c
        loss_s = self.args.style_weight * loss_s
        loss = loss_c + loss_s + (l_identity1 * 50) + (l_identity2 * 1)
        self.loss = loss

    def __inference__():
        assert (0.0 <= alpha <= 1.0)
        self.content_f, self.style_f = self.vgg(self.content, self.style)
        Fccc = self.sa_module(self.content_f, self.content_f)
        feat = self.sa_module(self.content_f, self.style_f)
        feat = feat * alpha + Fccc * (1 - alpha)
        return self.decoder(feat)
    def __backward__():
        self.loss.backward()
        self.optimizer.step()
        
    
    def __call__(self,content,style):
        self.content = content
        self.style = style
        self.__loss__()    
        self.__backward__()
        
    def summaryWriter(self,iteration,epoch):
        if iteration % arg.show_step == 0:
            Y = self.__inference__()
            image = utils.make_image(self.content, self.style, Y)
            
            writer.add_image('img', image)
            writer.add_scalar('training content loss',
                            self.loss,
                            epoch)