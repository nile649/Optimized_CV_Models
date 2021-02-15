import platform

ostype = platform.platform()[:7]
def dataiter(opt,size,content):
    if opt.device=='cuda' and ostype!="Windows":
        from .gpuDataLoader import dataCUDA
        return dataCUDA(opt,size,content)
    else:
        from .cpuDataLoader import dataCPU
        return dataCPU(opt,size,content)
