
# Adaptive-attention Completing Network (AACNet) for Remote Sensing Image
The code will be available.

<br>
This is the testing code for our AACNet to reconstruct remote sensing and natural image. Given one image and mask, the proposed **AACNet** model is able to reconstruct masked regions. This code is adapted from an initial fork of [PIC](https://github.com/lyndonzheng/Pluralistic-Inpainting) implementation.

## Illustration of Ada-attention
![](https://github.com/huangwenwenlili/AACNet/blob/master/images/ada-attention-results.png)

Illustration of self-attention and Ada-attention, including high attention score points, key/value positions, and inpainting results. The gray covering regions represent the corrupted regions. In the high attention score points images, the red stars show the specific query, and the purple circles show the high attention score points. These attention score images show that Ada-attention focuses more on relevant keys related to the query, e.g., the roof. In the key/value positions images, cyan dots denote the original uniform coordinates used in self-attention, and red dots of Ada-attention denote the sampled coordinates adjusted by the offset subnet, which are more inclined to the edge and texture with rich features. In the inpainting results images, local details are displayed in the red box, demonstrating that our Ada-attention generates superior results.


# Getting started
## Installation
This code was tested with Pytoch 1.8.1 CUDA 11.1, Python 3.6 and Ubuntu 18.04

- Create conda environment:

```
conda create -n inpainting-py36 python=3.6
conda deactivate
conda activate inpainting-py36
pip install visdom dominate
```
- Clone this repo:

```
git clone https://github.com/huangwenwenlili/AACNet
cd AACNet
```

- Pip install libs:

```
pip install -r requirements.txt
```

## Datasets
- ```AID```: It contains 30 scene categories of RS aerial RGB images. 10000 images. [AID](https://captain-whu.github.io/AID/) can download on [Onedrive](https://1drv.ms/u/s!AthY3vMZmuxChNR0Co7QHpJ56M-SvQ) or [BaiduPan](https://pan.baidu.com/s/1mifOBv6#list/path=%2F).
- ```PatternNet```: It contains 45 scene categories of RS digital images. 30400 images. [PatternNet](https://sites.google.com/view/zhouwx/dataset)
- ```NWPU-RESISC45```: It contains 38 scene categories of RS digital images. 31500 images. [NWPU-RESISC45](https://arxiv.org/abs/1703.00121)
- ```Paris StreetView```: It contains buildings of Paris of natural digital images. 14900 training images and 100 testing images. [Paris](https://github.com/pathak22/context-encoder)
- ```CelebA-HQ```: It contains celebrity face images. 30000 images. [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ)

## Testing

- Test the model. Input images and masks resolution are 256*256. In the testing, we use [irregular mask dataset](https://github.com/NVIDIA/partialconv) to evaluate different ratios of corrupted region images.

```
python test.py  --name aid --img_file your_image_path
```
- Set ```--mask_type``` in options/base_options.py to test various masks. ```--mask_file``` path is needed for **3. external irregular mask**,
- The default results will be saved under the *results* folder. Set ```--results_dir``` for a new path to save the result.

## Pretrained Models
Download the pre-trained models using the following links and put them under```checkpoints/``` directory.
BaiduPan link: https://pan.baidu.com/s/1OND83WeaViSUOGu7v_DJXg?pwd=skix extract code:skix

- ```RS model```: AID | PatternNet | NWPU-RESISC45
- ```Natural model```: Paris StreetView | CelebA-HQ

Our models are trained with images of resolution 256*256 with random regular and irregular holes.

## Example Results
- **Completion Results for RS Datasets**
![](https://github.com/huangwenwenlili/AACNet/blob/master/images/rs-results.png)

- **Completion Results for Natural Inpainting Datasets**
![](https://github.com/huangwenwenlili/AACNet/blob/master/images/natural-results.png)


## License
<br />
The codes and the pre-trained models in this repository are under the MIT license as specificed by the LICENSE file.
This code is for educational and academic research purpose only.

## Reference Codes
- https://github.com/lyndonzheng/Pluralistic-Inpainting
- https://github.com/NVIDIA/partialconv

## Citation

If you use this code for your research, please cite our paper.
```
@article{huang2023adaptive,
  title={Adaptive-attention Completing Network (AACNet) for Remote Sensing Image},
  author={Huang, Wenli and Deng, Ye and Hui, Siqi and Wang, Jinjun},
  journal={},
  volume={},
  pages={},
  year={}
}
```
