from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from random import shuffle
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import numpy as np
class ExternalInputIterator(object):
    def __init__(self, batch_size=1, content_img=''):
        self.content = content_img
        self.batch_size = batch_size

    def __iter__(self):
        self.i = 0
        self.n = len(self.content)
        return self

    def __next__(self):
        batch_content = []
        # for _ in range(self.batch_size):
        f_c = self.content
        f_c = open(f_c, 'rb')
        batch_content.append(np.frombuffer(f_c.read(), dtype = np.uint8))
        # self.i = (self.i + 1) % self.n
        return (batch_content)
          
    next = __next__


class ExternalSourcePipeline(Pipeline):
    def __init__(self, data_iterator, batch_size,size, num_threads, device_id):
        super(ExternalSourcePipeline, self).__init__(batch_size,
                                      num_threads,
                                      device_id,
                                      seed=12)
        self.data_iterator = data_iterator
        self.ct = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB,)
        self.cast = ops.Cast(device = "gpu",
                             dtype = types.INT32)

        self.resize = ops.Resize(device="gpu", resize_x=size, resize_y=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            mean=[0, 0, 0],
                                            std=[255.0, 255.0, 255.0]
                                            )
 
    
    def define_graph(self):
        self.jpegs_ct = self.ct()
        images_ct = self.decode(self.jpegs_ct)
        output_ct = self.resize_(images_ct)
        return (self.cmnp(output_ct))

    def iter_setup(self):

        # the external data iterator is consumed here and fed as input to Pipeline
        ct = self.data_iterator.next()
        self.feed_input(self.jpegs_ct, ct)



def dataCUDA(opt,size,content):
    eii = ExternalInputIterator(batch_size=opt.batch_size, 
    content_img=content)
    iterator = iter(eii)
    pipe = ExternalSourcePipeline(data_iterator=iterator, batch_size=opt.batch_size,size=size, num_threads=opt.num_workers, device_id=0)
    pipe.build()
    print("DALI INITIATED")
    data_iter = DALIGenericIterator([pipe], ['image'],dynamic_shape=True,size=1, auto_reset=False)
    return data_iter

