from ImgEncry import utils
from ImgEncry.ImgEncry import Encrypt
from ImgEncry import methods
from PIL import Image
import timeit
import argparse

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default = 0, help="[methods.encSubImage,methods.encRowColShuffleImage,methods.encKeyImage")
    parser.add_argument("--image",type=str, default = "./Amber-Heard-Johnny-Depp.png")
    opt = parser.parse_args()
    return opt



if __name__ == "__main__":
	arg = get_opt()
	__Methods__ = [methods.encSubImage,methods.encRowColShuffleImage,methods.encKeyImage]
	obj = Encrypt(utils,__Methods__[arg.m])

	tic = timeit.default_timer()
	img = obj(Image.open(arg.image).convert("RGB"),"hola",1)
	tac = timeit.default_timer()
	print("{} seconds to encrypt the image".format(tac-tic))
	img.save('encrypted.png')

	tic = timeit.default_timer()
	img = obj(img,"hola",0)
	tac = timeit.default_timer()
	print("{} seconds to decrypt the image".format(tac-tic))
	img.save('decrypted.png')