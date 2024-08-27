# This is the code of paper "ASCNet: Asymmetric Sampling Correction Network for Infrared Image Destriping", the full code will be made public after the manuscript is accepted.
[[Paper]](https://arxiv.org/abs/2401.15578)
# Chanlleges and inspiration   
![Image text](https://github.com/xdFai/ASCNet/blob/main/Fig/Fig0.png)

# Structure
![Image text](https://github.com/xdFai/ASCNet/blob/main/Fig/Fig2.png)

![Image text](https://github.com/xdFai/ASCNet/blob/main/Fig/Fig3.png)


# Examples

[<img src="https://github.com/xdFai/ASCNet/blob/main/Fig/Mars.png" width="385">](https://imgsli.com/MjkxNDU2) | [<img src="https://github.com/xdFai/ASCNet/blob/main/Fig/Building.png" width="385">](https://imgsli.com/MjkxNDU4)
:-------------------------:|:-------------------------:
Mars | Building


[<img src="https://github.com/xdFai/ASCNet/blob/main/Fig/Road.png" width="385">](https://imgsli.com/MjkxNDU5) | [<img src="https://github.com/xdFai/ASCNet/blob/main/Fig/Car.png" width="385">](https://imgsli.com/MjkxNDYx)
:-------------------------:|:-------------------------:
Road | Car 

## Usage

#### 1. Dataset
Training dataset: [[Data]](https://drive.google.com/file/d/1o9BmWspPTJtFsBj66NN3FfM83cjp37IW/view?usp=sharing)

Training dataset augmentation: [[Data_AUG]](https://drive.google.com/file/d/1Iv4CoQiInFORYn1kHjJCCCeuy6LKvnIc/view?usp=sharing)


##### 2. Train.
```bash
python train.py
```

#### 3. Test and demo.
```bash
python test.py
```

## Contact
**Welcome to raise issues or email to [yuansy@stu.xidian.edu.cn](yuansy@stu.xidian.edu.cn) or [yuansy2@student.unimelb.edu.au](yuansy2@student.unimelb.edu.au) for any question regarding our ASCNet.**
