# Semantic-segmentation-framework
This is a semantic segmentation framework, it can be easy to read and learn. It also can add your network and dataset to train. 

### 0. Quick start

1. Git clone from GitHub.

```
git clone https://github.com/ShanZard/Semantic-segmentation-framework.git
```

2. cd to Semantic-segmentation-framework

```sh
cd Semantic-segmentation-framework
```

3. install requirements

```
pip install -r requirements
```



### 1. Dataset

1. Should organize dataset like this:

```
├── data
│   ├── your_dataset
|   |   |——train
|   |   |   ├── images
│   │   │   │   ├── xxx{img_suffix}
│   │   │   │   ├── yyy{img_suffix}
│   │   │   │   ├── zzz{img_suffix}
│   │   │   ├── mask
│   │   │   │   ├── xxx{seg_map_suffix}
│   │   │   │   ├── yyy{seg_map_suffix}
│   │   │   │   ├── zzz{seg_map_suffix}
│   │   ├── val/test
│   │   │   ├── images
│   │   │   │   ├── xxx{img_suffix}
│   │   │   │   ├── yyy{img_suffix}
│   │   │   │   ├── zzz{img_suffix}
│   │   │   ├── mask
│   │   │   │   ├── xxx{seg_map_suffix}
│   │   │   │   ├── yyy{seg_map_suffix}
│   │   │   │   ├── zzz{seg_map_suffix}
```

Note: 

1. images names should be same as mask
1. mask should be [h,w], if your mask is RGB[3,h,w],  you can use ```tools/pre_processdataset.py``` to convert it .

### 2. Before train

1. should change ```utils/palette.py```
2. change your args in ```train.py```

### 3. Test

