import numpy as np
import math
class utils:
    def __init__(self):
        pass
    
    def __BreakImg__(self,img,slice_size=100):

        """slice an image into parts slice_size tall"""
        listImg = []
        width, height = img.size
        upper = 0
        left = 0
        slices = int(math.ceil(height/slice_size))

        count = 1
        for slice in range(slices):
            #if we are at the end, set the lower bound to be the bottom of the image
            if count == slices:
                lower = height
            else:
                lower = int(count * slice_size)  
            #set the bounding box! The important bit     
            bbox = (left, upper, width, lower)
            working_slice = img.crop(bbox)
            listImg.append((count,working_slice))
            upper += slice_size
            count +=1
        return listImg
    
    def __AttachImg__(self,imgList,height,width):
        new_im = Image.new('RGB', (height, width))
        x_offset = 0
        for im in images:
            new_im.paste(im, (1,x_offset))
            x_offset += im.size[1]
        return new_im
    
    def __SetSeed__(self,password):
        passValue=0
        for ch in password:                 
            passValue=passValue+ord(ch)
        np.random.seed(passValue)