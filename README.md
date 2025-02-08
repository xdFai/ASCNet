# This is the code of paper "ASCNet: Asymmetric Sampling Correction Network for Infrared Image Destriping".[[Paper]](https://ieeexplore.ieee.org/document/10855453) [[Weight]](https://drive.google.com/file/d/1zbBsWUbRVBjNckPg5DiCgKIKOKWnQ2N8/view?usp=sharing)
Shuai Yuan, Hanlin Qin, Xiang Yan, Shiqi Yang, Shuowen Yang, Naveed Akhtar, Huixin Zhou, IEEE Transactions on Geoscience and Remote Sensing 2025.
# Real Destriping Examples

[<img src="https://github.com/xdFai/ASCNet/blob/main/Fig/Mars.png" width="385">](https://imgsli.com/MjkxNDU2) | [<img src="https://github.com/xdFai/ASCNet/blob/main/Fig/Building.png" width="385">](https://imgsli.com/MjkxNDU4)
:-------------------------:|:-------------------------:
Mars | Building


[<img src="https://github.com/xdFai/ASCNet/blob/main/Fig/Road.png" width="385">](https://imgsli.com/MjkxNDU5) | [<img src="https://github.com/xdFai/ASCNet/blob/main/Fig/Car.png" width="385">](https://imgsli.com/MjkxNDYx)
:-------------------------:|:-------------------------:
Road | Car 


# Chanlleges and inspiration   
![Image text](https://github.com/xdFai/ASCNet/blob/main/Fig/Fig0.png)

# Structure
![Image text](https://github.com/xdFai/ASCNet/blob/main/Fig/Fig2.png)

![Image text](https://github.com/xdFai/ASCNet/blob/main/Fig/Fig3.png)

# If the implementation of this repo is helpful to you, just star it！⭐⭐⭐

## Usage

#### 1. Dataset
Training dataset: [[Data]](https://drive.google.com/file/d/1o9BmWspPTJtFsBj66NN3FfM83cjp37IW/view?usp=sharing)

Training dataset augmentation: [[Data_AUG]](https://drive.google.com/file/d/1Iv4CoQiInFORYn1kHjJCCCeuy6LKvnIc/view?usp=sharing)


##### 2. Train.
```bash
python train.py
```

#### 3. Test and demo. [[Weight]](https://drive.google.com/file/d/1zbBsWUbRVBjNckPg5DiCgKIKOKWnQ2N8/view?usp=sharing)
```bash
python test.py
```

If you find the code useful, please consider citing our paper using the following BibTeX entry.

```
@ARTICLE{10855453,
  author={Yuan, Shuai and Qin, Hanlin and Yan, Xiang and Yang, Shiqi and Yang, Shuowen and Akhtar, Naveed and Zhou, Huixin},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={ASCNet: Asymmetric Sampling Correction Network for Infrared Image Destriping}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Noise;Discrete wavelet transforms;Semantics;Image reconstruction;Feature extraction;Neural networks;Filters;Crosstalk;Aggregates;Geoscience and remote sensing;Infrared image destriping;deep learning;asymmetric sampling;wavelet transform;column correction},
  doi={10.1109/TGRS.2025.3534838}}
```

## Contact
**Welcome to raise issues or email to [yuansy@stu.xidian.edu.cn](yuansy@stu.xidian.edu.cn) or [yuansy2@student.unimelb.edu.au](yuansy2@student.unimelb.edu.au) for any question regarding our ASCNet.**
