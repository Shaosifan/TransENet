# TransENet for remote sensing image super-resolution
Official Pytorch implementation of the paper "[Transformer-based Multi-Stage Enhancement for Remote Sensing Image Super-Resolution](https://ieeexplore.ieee.org/document/9654169)" accepted by IEEE TGRS.  

Convolutional neural networks have made great breakthrough in recent remote sensing image super-resolution tasks. Most of these methods adopt upsampling layers at the end of the models to perform enlargement, which ignores feature extraction in the high-dimension space and thus limits super-resolution performance. To address this problem, we propose a new super-resolution framework for remote sensing image to enhance the high-dimensional feature representation after the upsampling layers. We name the proposed method as Transformer-based Enhancement Network (TransENet), where transformers are introduced to exploit features at different levels. The core of the TransENet is a transformer-based multi-stage enhancement structure which can be combined with traditional super-resolution frameworks to fuse multi-scale high/low-dimension features. Specifically, in this structure, the encoders aim to embed the multi-level features in the feature extraction part and the decoders are used to fuse these encoded embeddings. Experimental results demonstrate that our proposed TransENet can improve super-resolved results and obtain superior performance over several state-of-the-art methods.

## Requirements
- Python 3.6+
- Pytorch>=1.6
- torchvision>=0.7.0
- einops
- matplotlib
- cv2
- scipy
- tqdm
- scikit


## Installation
Clone or download this code and install aforementioned requirements 
```
cd codes
```

## Train
Download the [UCMerced dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html) and [AID dataset](https://captain-whu.github.io/AID/) and split them into train/val/test data, where the original images would be taken as the HR references and the corresponding LR images are generated by bicubic down-sample. 
```
# x4
python demo_train.py --model=TRANSENET --dataset=UCMerced --scale=4 --patch_size=192 --ext=img --save=TRANSENETx4_UCMerced
# x3
python demo_train.py --model=TRANSENET --dataset=UCMerced --scale=3 --patch_size=144 --ext=img --save=TRANSENETx4_UCMerced
# x2
python demo_train.py --model=TRANSENET --dataset=UCMerced --scale=2 --patch_size=96 --ext=img --save=TRANSENETx4_UCMerced
```

The train/val data pathes are set in [data/__init__.py](codes/data/__init__.py) 

## Test 
The test data path and the save path can be edited in [demo_deploy.py](codes/demo_deploy.py)

```
# x4
python demo_deploy.py --model=TRANSENET --scale=4
# x3
python demo_deploy.py --model=TRANSENET --scale=3
# x2
python demo_deploy.py --model=TRANSENET --scale=2
```

## Evaluation 
Compute the evaluated results in term of PSNR and SSIM, where the SR/HR paths can be edited in [calculate_PSNR_SSIM.py](codes/metric_scripts/calculate_PSNR_SSIM.py)

```
cd metric_scripts 
python calculate_PSNR_SSIM.py
```

## Citation 
If you find this code useful for your research, please cite our paper:
``````
@article{lei2021transformer,
  title={Transformer-based Multi-Stage Enhancement for Remote Sensing Image Super-Resolution},
  author={Lei, Sen and Shi, Zhenwei and Mo, Wenjing},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2021},
  publisher={IEEE}
}
``````

## Acknowledgements 
This code is built on [RCAN (Pytorch)](https://github.com/yulunzhang/RCAN) and [EDSR (Pytorch)](https://github.com/sanghyun-son/EDSR-PyTorch). We thank the authors for sharing the codes.  

