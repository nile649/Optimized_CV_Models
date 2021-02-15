Code is directly copied from with wrappers to make it look cool, we also use feathering code from block-shuffle mainly.
1. https://github.com/mumair5393/Style-Transfer-with-Style-Attentional-Networks
2. https://github.com/ProGamerGov/Neural-Tile
3. https://github.com/czczup/block-shuffle

conda create --name test_st_38 python=3.8
conda/source activate test_st_38 

nvcc --version

According to nvcc Install Pytorch and Nvidia-Dali from their respective website.

https://pytorch.org/

https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html
[Dali is only supported on Linux machines]

conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch

pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100

pyheif==0.5.1 "only for linux"


pip install -r requirements.txt