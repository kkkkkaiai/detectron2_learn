# detectron2_learn

**Environment**

| Item          |   Content   |
| ------------- | :---------: |
| OS            | Ubuntu18.04 |
| Nvidia-driver | 450.102.04  |
| Cuda          |    10.1     |
| Cudnn         |  7.6.5.32   |

## Detectron2 安装方法

**install torch**

```bash
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

**install detectron**

```bash
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
```

**install dependency**
```bash
pip install opencv-python
pip install labelme
```

**为jupyter notebook创建kernel**

```bash
pip install ipykernel
# create ipykernel env
python -m ipykernel install --user --name detectron2
```



## Project

### Workpieces

该project中示例了如何使用detectron2进行基于coco的目标检测。

#### 1.首先使用labelme打标签

将生成的文件按照该目录所示的方式进行布置。

```
├── DIR
│  ├── images
│  │  ├── 1.jpg
│  │  ├── ...
│  │  ├── *.jpg
│  ├── 1.json
│  ├── ...
└──└── *.json
```

#### 2.将多个COCO格式的json文件合成一个文件

在本示例中，将合成的json文件命名为trainval.json。

`labelme2coco_object_detection.ipynb` 用于在labelme中标记为rectangle的标签文件进行合并。

#### 3.训练

在合成标签后，使用`train_test.ipynb`进行训练。

需要修改以下变量：

`project`      项目的名称

`yaml_file`  yaml文件的位置

`weights `      所选择的模型文件

#### 4.测试（cam）

在本示例中，使用网络相机进行测试。该方法改自于`detectron2`的[`demo.py`](https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py)。

### Toy_workpiece
#### 1.首先使用labelme打标签

将生成的文件按照该目录所示的方式进行布置。

```
├── DIR
│  ├── images
│  │  ├── 1.jpg
│  │  ├── ...
│  │  ├── *.jpg
│  ├── 1.json
│  ├── ...
└──└── *.json
```

#### 2.将多个COCO格式的json文件合成一个文件

在本示例中，将合成的json文件命名为trainval.json。

`labelme2coco_instance.ipynb` 用于在labelme中标记为polygons的标签文件进行合并。

#### 3.训练

在合成标签后，使用`train_test.ipynb`进行训练。

需要修改以下变量：

`project`      项目的名称

`yaml_file`  yaml文件的位置

`weights `      所选择的模型文件

#### 4.测试（cam）

在本示例中，使用网络相机进行测试。该方法改自于`detectron2`的[`demo.py`](https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py)。



