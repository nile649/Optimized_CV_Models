import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
import os
import numpy as np
from .DataTransform import Transform
import pdb       

class DataLoader(data.Dataset):
    ''' 

    Create Pytorch Custom dataloader to load cpu Dataset

    '''
    def __init__(self,content,size):
        super(DataLoader, self).__init__()
        self.content = content
        transform = Transform()
        self.transform = transform(size)

    def __getitem__(self, index):
        content = self.content#[index]
        content = Image.open(content).convert('RGB')
        content_ = self.transform(content)
        return content_

    def __len__(self):
        return len(self.content)

    def name(self):
        return 'Single Image Dataset Loader'
     
def dataCPU(opt,size,content):
    dataset = DataLoader(content,size)
    return iter(data.DataLoader(
            dataset, batch_size=opt.batch_size,
            num_workers=opt.num_workers))
    print("PYTORCH INITIATED")