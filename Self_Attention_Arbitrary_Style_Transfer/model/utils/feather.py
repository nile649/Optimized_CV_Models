from abc import abstractmethod
import torch.nn as nn
import math
import torch
import torch.nn.functional as F

class parametersFeather(type):
    def __init__(self, name, bases, attrs):
        '''
        This parameters are inherited to each object.
        
        '''
        self.padding_width = None
        self.basic_width = None
        self.block_width = None
        self.blending_width = None
        self.width = None
        self.height = None
        self.width_expand = None
        self.height_expand = None
        self.og_width = None
        self.og_height = None
        self.device = None
        self.squareSz = False

class MainFeather(metaclass=parametersFeather):
    def __init__(self):
        pass
    
    @classmethod
    def set_IMG_Attr(ctx,**kwargs):
        for arg in kwargs.items():
            if hasattr(ctx, arg[0]):
                setattr(ctx,arg[0],arg[1]) 
    @classmethod
    @abstractmethod            
    def breakImg(self,img):
        pass
    
    @classmethod
    @abstractmethod
    def attachImg(self,img):
        pass
    
    @classmethod
    @abstractmethod
    def restore(ctx,block_list):
        pass
    
    @staticmethod
    def unpadding(image,padding):
        width, height = image.shape[2], image.shape[3]
        image = image[:,:,padding:height - padding, padding:width - padding]
        return image
    
    @classmethod
    def padding(ctx,image):
        m = nn.ReflectionPad2d(ctx.padding_width)
        new_image = m(image)
        return new_image  
    @classmethod
    def cut(ctx,image):
        row_num = math.ceil(ctx.height / ctx.basic_width)
        col_num = math.ceil(ctx.width / ctx.basic_width)  # 行数和列数
        image_list = []
        for j in range(0, row_num):
            b = j * ctx.basic_width
            d = (j + 1) * ctx.basic_width + 2 * ctx.padding_width
            for i in range(0, col_num):
                a = i * ctx.basic_width
                c = (i + 1) * ctx.basic_width + 2 * ctx.padding_width
                image_block = image[:,:,b:d, a:c]
                image_list.append(image_block)
        return image_list   
    

class Feather_(MainFeather):
    def __init__(self):
        super(Feather_, self).__init__()
        pass
    
    def breakImg(ctx,image):
        return ctx.cut(Feather_.padding(image))
        
    def attachImg(ctx,block_list):
        if ctx.squareSz:
            return ctx.unpadding(ctx.restore(block_list), ctx.padding_width)
        else:
            return F.interpolate(ctx.unpadding(ctx.restore(block_list), ctx.padding_width),\
                            size=(ctx.og_width,ctx.og_height))
        
        
    def restore(ctx,block_list):
        def horizontal_blend(overlap1, overlap2):
    #         pdb.set_trace()
            overlap1 = overlap1
            overlap2 = overlap2
            _ = torch.tensor([[[[i/(2*ctx.blending_width) for i in range(2*ctx.blending_width)]]]])
            horizontal_alpha = _.repeat(1,3,overlap1.shape[2],1).to(ctx.device)#.cuda()
            target = (overlap1 * (1 - horizontal_alpha) + overlap2 * horizontal_alpha)#.astype(np.uint8)
            return target

        def vertical_blend(overlap1, overlap2):
            overlap1 = overlap1
            overlap2 = overlap2
            _ = torch.tensor([[[[i/(2*ctx.blending_width)] for i in range(2*ctx.blending_width)]]])
            vertical_alpha = _.repeat(1,3,1,ctx.width_expand).to(ctx.device)#.cuda()
            target = (overlap1 * (1 - vertical_alpha) + overlap2 * vertical_alpha)#.astype(np.uint8)
            return target

        row_num = math.ceil(ctx.height / ctx.basic_width)
        col_num = math.ceil(ctx.width / ctx.basic_width)
        block_list = [block_list[i:i+col_num] for i in range(0, len(block_list), col_num)]
        row_images = []
#         pdb.set_trace()
        for i, row in enumerate(block_list):
            row_image = None
            for j, item in enumerate(row):
                if j == 0:
                    row_image = item
                else:
                    if ctx.blending_width != 0:
                        left = row_image[:,:,:, 0:-2*ctx.blending_width]
                        overlap1 = row_image[:,:,:, -2*ctx.blending_width:]
                        overlap2 = item[:,:,:, 0:2*ctx.blending_width]
                        right = item[:,:,:, 2*ctx.blending_width:]
                        
                        overlap = horizontal_blend(overlap1, overlap2)
    #                     row_image = cv2.hconcat([left, overlap, right])
                        row_image = torch.cat([left, overlap, right],3)
                    else:
                        row_image = torch.cat([row_image, item],3)
            row_images.append(row_image)

        image = None
        for i, row in enumerate(row_images):
            if i == 0:
                image = row
            else:
                if ctx.blending_width != 0:
                    top = image[:,:,0:-2*ctx.blending_width, :]
                    overlap1 = image[:,:,-2*ctx.blending_width:, :]
                    overlap2 = row[:,:,0:2*ctx.blending_width, :]
                    bottom = row[:,:,2*ctx.blending_width:, :]
                    overlap = vertical_blend(overlap1, overlap2)
    #                 image = cv2.vconcat([top, overlap, bottom])
                    image = torch.cat([top, overlap, bottom],2)
                else:
                    image = torch.cat([image, row],2)

        return image
