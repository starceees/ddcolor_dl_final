# ddcolor_dl_final

# DD-Color
<p align="center">
  <img src="assets/network_arch.jpg" width="100%">
</p>

🪄 DDColor can provide vivid and natural colorization for historical black and white old photos.
🎲 It can even colorize/recolor landscapes from anime games, transforming your animated scenery into a realistic real-life style! (Image source: Genshin Impact)
##Inputs
<table>
  <tr>
    <td>
      <img src="assets/test_images/Ansel Adams _ Moore Photography.jpeg" alt="Image 1" width="200">
    </td>
    <td>
      <img src="assets/test_images/Detroit circa 1915.jpg" alt="Image 2" width="200">
    </td>
    <td>
      <img src="assets/test_images/New York Riverfront December 15, 1931.jpg" alt="Image 3" width="200">
    </td>
    <td>
      <img src="assets/test_images/Einstein, Rejection, and Crafting a Future.jpeg" alt="Image 4" width="200">
    </td>
  </tr>
</table>
##Ouputs

# Pre-testing
DDCOlor uses ConvNext-Net Pretrained weights for the backbone of the encoder to extract the image features. Download them from the link.
Download pretrained weights for [ConvNeXt](https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth) and [InceptionV3](https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth) and put it into `pretrain` folder.

# Implementation 

### Requirements

- Python >= 3.8
- PyTorch >= 1.7

### Install with conda (Recommend)
The repository for DDColor suggests runnign with python 3.9, But we found usage with python 3.10.12 was smoother.
```
conda create -n ddcolor python=3.10
conda activate ddcolor
pip install -r requirements.txt

python3 setup.py develop  # install basicsr
```
# Value Params 

## Acknowledgments
We thank the authors of BasicSR for the awesome training pipeline.

> Xintao Wang, Ke Yu, Kelvin C.K. Chan, Chao Dong and Chen Change Loy. BasicSR: Open Source Image and Video Restoration Toolbox. https://github.com/xinntao/BasicSR, 2020
