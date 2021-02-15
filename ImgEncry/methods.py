from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from random import randrange, seed
import os
import pdb
import copy


class encRowColShuffleImage:
    '''
    code : https://codegolf.stackexchange.com/questions/35005/
    rearrange-pixels-in-image-so-it-cant-be-recognized-and-then-get-it-back
    
    '''
    def __init__(self,utils):
        self.utils = utils()
        
    def __ScrambleColumns__(self,im,columns,rows):
        pixels =list(im.getdata())
        newOrder=list(range(columns))     
        np.random.shuffle(newOrder)            #shuffle

        newpixels=[]
        for i in range(rows):
            for j in range(columns):
                newpixels+=[pixels[i*columns+newOrder[j]]]

        im.putdata(newpixels)
        return im

    def __UnscrambleColumns__(self,im,columns,rows):
        pixels =list(im.getdata())

        newOrder=list(range(columns))     
        np.random.shuffle(newOrder)            #shuffle

        newpixels=copy.deepcopy(pixels)
        for i in range(rows):
            for j in range(columns):
                newpixels[i*columns+newOrder[j]]=pixels[i*columns+j]

        im.putdata(newpixels)
        return im

    def __ScrambleRows__(self,im,columns,rows):
        pixels =list(im.getdata())

        newOrder=list(range(rows))        
        np.random.shuffle(newOrder)            #shuffle the order of pixels

        newpixels=copy.deepcopy(pixels)
        for j in range(columns):
            for i in range(rows):
                newpixels[i*columns+j]=pixels[columns*newOrder[i]+j]

        im.putdata(newpixels)
        return im

    def __UnscrambleRows__(self,im,columns,rows):
        pixels =list(im.getdata())

        newOrder=list(range(rows))        
        np.random.shuffle(newOrder)            #shuffle the order of pixels

        newpixels=copy.deepcopy(pixels)
        for j in range(columns):
            for i in range(rows):
                newpixels[columns*newOrder[i]+j]=pixels[i*columns+j]

        im.putdata(newpixels)
        return im


    #set random seed based on the given password

    def __Encrypt__(self,im,columns,rows,password):
        self.utils.__SetSeed__(password)
        if im.size[0]%2==0:
            # scramble(im,columns,rows)
            return self.__ScrambleColumns__(im,columns,rows)
        else:
            return self.__ScrambleRows__(im,columns,rows)

    def __Decrypt__(self,im,columns,rows,password):
        self.utils.__SetSeed__(password)
        if im.size[0]%2==0:
            # unscramble(im,columns,rows)
            self.__UnscrambleColumns__(im,columns,rows)
        else:
            self.__UnscrambleRows__(im,columns,rows)
        return im
    
    def __call__(self,img,coulmns,rows,password,enc=1):
        
        if enc==1:
            return self.__Encrypt__(copy.deepcopy(img),coulmns,rows,password)
        else:
            return self.__Decrypt__(copy.deepcopy(img),coulmns,rows,password)

class encSubImage:
    def __init__(self,utils):
        self.utils = utils()
    
    def __Encrypt__(self,img,coulmns,rows,password="hello world"):
        print(password)
        self.utils.__SetSeed__(password)
        keyImg = np.floor(np.random.rand(img.size[1],img.size[0],3) * 11)*10
        temp = np.asarray(img)
        res = img-2*keyImg
        return Image.fromarray(res.astype('uint8')).convert('RGB')
    def __Decrypt__(self,img,coulmns,rows,password):
        self.utils.__SetSeed__(password)
        keyImg = np.floor(np.random.rand(img.size[1],img.size[0],3) * 11)*10
        res = img+2*keyImg
        return Image.fromarray(res.astype('uint8')).convert('RGB')
    def __call__(self,img,coulmns,rows,password,enc=1):
        if enc==1:
            return self.__Encrypt__(img,coulmns,rows,password)
        else:
            return self.__Decrypt__(img,coulmns,rows,password)
        

class encKeyImage:
    def __init__(self,utils):
        self.utils = utils()

    def __Encrypt__(self,input_image,password="hello world",number_of_regions=16777216):
        self.utils.__SetSeed__(password)
        
        
        if input_image.size == (1, 1):
            raise ValueError("input image must contain more than 1 pixel")
        number_of_regions = min(int(number_of_regions),
                                self.__NumberOfCcolours__(input_image))
        
        imarray = np.floor(np.random.rand(input_image.size[1],input_image.size[0],3) * 11)*10
        key_image = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        
#         key_image = Image.open('./nami.jpg').convert("RGB")
        
        region_lists = self.__CreateRegionLists__(input_image, key_image,
                                         number_of_regions)
        seed(120)
        self.__Shuffle__(region_lists)
        output_image = self.__SwapPixels__(input_image, region_lists)
        return output_image


    def __NumberOfCcolours__(self,image):
        return len(set(list(image.getdata())))


    def __CreateRegionLists__(self,input_image, key_image, number_of_regions):
        template = self.__CreateTemplate__(input_image, key_image, number_of_regions)
        number_of_regions_created = len(set(template))
        region_lists = [[] for i in range(number_of_regions_created)]
        for i in range(len(template)):
            region = template[i]
            region_lists[region].append(i)
        odd_region_lists = [region_list for region_list in region_lists
                            if len(region_list) % 2]
        for i in range(len(odd_region_lists) - 1):
            odd_region_lists[i].append(odd_region_lists[i + 1].pop())
        return region_lists


    def __CreateTemplate__(self,input_image, key_image, number_of_regions):
        if number_of_regions == 1:
            width, height = input_image.size
            return [0] * (width * height)
        else:
            resized_key_image = key_image#.resize(input_image.size, Image.NEAREST)
            pixels = list(resized_key_image.getdata())
            pixel_measures = [self.__Measure__(pixel) for pixel in pixels]
            distinct_values = list(set(pixel_measures))
            number_of_distinct_values = len(distinct_values)
            number_of_regions_created = min(number_of_regions,
                                            number_of_distinct_values)
            sorted_distinct_values = sorted(distinct_values)
            while True:
                values_per_region = (number_of_distinct_values /
                                     number_of_regions_created)
                value_to_region = {sorted_distinct_values[i]:
                                   int(i // values_per_region)
                                   for i in range(len(sorted_distinct_values))}
                pixel_regions = [value_to_region[pixel_measure]
                                 for pixel_measure in pixel_measures]
                if self.__NoSmallPixelRegions__(pixel_regions,
                                          number_of_regions_created):
                    break
                else:
                    number_of_regions_created //= 2
            return pixel_regions


    def __NoSmallPixelRegions__(self,pixel_regions, number_of_regions_created):
        counts = [0 for i in range(number_of_regions_created)]
        for value in pixel_regions:
            counts[value] += 1
        if all(counts[i] >= 256 for i in range(number_of_regions_created)):
            return True


    def __Shuffle__(self,region_lists):
        for region_list in region_lists:
            length = len(region_list)
            for i in range(length):
                j = np.random.randint(length)
                region_list[i], region_list[j] = region_list[j], region_list[i]


    def __Measure__(self,pixel):
        '''Return a single value roughly measuring the brightness.

        Not intended as an accurate measure, simply uses primes to prevent two
        different colours from having the same measure, so that an image with
        different colours of similar brightness will still be divided into
        regions.
        '''
        if type(pixel) is int:
            return pixel
        else:
            r, g, b = pixel[:3]
            return r * 5879 + g * 3848 + b * 89


    def __SwapPixels__(self,input_image, region_lists):
        pixels = list(input_image.getdata())
        for region in region_lists:
            for i in range(0, len(region) - 1, 2):
                pixels[region[i]], pixels[region[i+1]] = (pixels[region[i+1]],
                                                          pixels[region[i]])
        scrambled_image = Image.new(input_image.mode, input_image.size)
        scrambled_image.putdata(pixels)
        return scrambled_image

    def __call__(self,img,coulmns,rows,password,enc=1):
        
        return self.__Encrypt__(copy.deepcopy(img),password)

