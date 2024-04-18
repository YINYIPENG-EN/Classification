# 环境说明

torch==1.7

torchvision==0.8.0

opencv==4.4.0.44

numpy==1.21.6

loguru==0.5.3

pandas==1.1.4

Pillow==9.5.0

tensorboardX==2.6.2.2

tqdm>=4.62.2

matplotlib==3.4.3

# 权重和数据集下载

这里数据集以猫狗数据集为例。

百度网盘：

链接：https://pan.baidu.com/s/1K_t-wB9-kvp29hQZyUau-A 
提取码：yypn 

数据集名称为catVSdog，下载数据集后解压会得到一个train和tes文件，里面分别存放了cat和dog**数据集图片**。

mobilenet_v2-b0353104.pth是ImageNet的预权重，mobilenet_catvsdog.pth是已经训练好的猫狗权重

将train和test放在datasets文件下，目录格式如下：

```
$datasets
|-- test
    |-- cat
    |-- dog
|-- train
    |-- cat
    |-- dog
```

权重放置在model_data中



# 训练自己的数据集

这里以猫狗数据集为例。

**1.检查数据集存放格式是否正确。存放格式：**

```
$datasets
|-- test
    |-- cat
    |-- dog
|-- train
    |-- cat
    |-- dog
```



**2.是否下载预权重并放入model_data文件下**

```
model_data
|-- cls_classes.txt
|-- mobilenet_catvsdog.pth
|-- mobilenet_v2-b0353104.pth
`-- vit-patch_16.pth

```

**3.创建自己类别的txt文件并放入model_data中，例如自己在model_data中新建一个cls_classes.txt内容为：**

```
cat
dog
```

**4.生成对应数据集标签的txt文件**

修改txt_annotation.py中的classes信息，要求和步骤3中提到的model_data/下的cls_classes.txt一致。

```
classes = ["cat", "dog"]
sets    = ["train", "test"]
```

修改并允许后会在项目中得到cls_train.txt和cls_test.txt两个文件。里面为标签文件信息。示例如下：

0和1分别对应猫和狗的标签，后面则是图像的路径。

```
0;E:\classification-pytorch-main/datasets/train\cat\cat.1.jpg
1;E:\classification-pytorch-main/datasets/train\dog\dog.9966.jpg
```

**5.输入命令开启训练：**

```bash
python train.py --classes_path model_data/cls_classes.txt model_data/mobilenet_v2-b0353104.pth --batch_size 16
```

传入参数说明：根据自己的需求开启训练

--cuda:是否采用GPU训练，默认为True

--classes_path: classes txt path

--input_shape:输入大小，默认为224x224

--backbone:默认为mobilenet,支持vit,resnet50,vgg16,alexnet

--model_path:预训练权重路径

--Freeze:是否开启冻结训练，默认为True

--annatation_path:训练集的txt路径，默认为cls_train.txt

--num_wokers:线程数量，默认为4

--init_epoch:初始的epoch，默认为0，表示从第0个epoch开始

--Freeze_epoch：冻结训练的epoch数量，默认30，表示前30个epoch为冻结训练

--epoch:总的epoch数量，默认为100

--lr:初始学习率，默认为1e-3

--batch_size：batch size大小，根据自己的显存情况设置batch 大小

根据参数传入进行训练，当出现下面打印说明训练已经开始：

```
num_classes:2
device_ids:  [0]
Loading weights into state dict...
<All keys matched successfully>
number of train datasets:  18000
number of val datasets:  2000

start freeze train

2024-04-17 22:19:34.595 | INFO     | utils.utils_fit:fit_one_epoch:25 - Start Train
Epoch 1/30:   1%|█                                                                                          | 55/4500 [00:32<44:16,  1.67it/s, accuracy=0.827, lr=0.001, total_loss=0.465]
```



训练期间的**tensorboard**和**ROC**以及模型会保存在logs文件夹中

## tensorboard的使用

在使用tensorboard前请确保你已安装相关环境。

在训练期间或者训练后可以利用tensorboard实时观看训练中的ACC、loss、lr等。

训练中的tensorboard相关信息会保存在logs/tensorboard_logs文件下，文件名为events.out.tfevents.xxxx的格式。

进入logs/，打开cmd(激活你的pytorch)，然后输入以下命令开启tensorboard。

```
tensorboard --logdir=./
```

输入命令后将会出现以下信息：

此时我们只需要打开浏览器，在浏览器输入下面的网址http://localhost:6006即可

```
TensorFlow installation not found - running with reduced feature set.
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.4.1 at http://localhost:6006/ (Press CTRL+C to quit)
```

# 图像检测

当我们训练好后就可以开启图像测试了，命令如下：

```
python detect.py --weights your weights path --model_name mobilenet --img_path your image path
```

检测后图像将会保存在**output**文件下



# 测试

比如要测试一下top-1和top-5的acc测试结果，只需要运行以下命令即可，以top-1为例：

```
python eval_top1.py --weights your weight path
```



# 特征可视化

本项目还新增了在检测中进行特征可视化功能，用于模型的分析。

```
python detect.py --weights 权重路径 --img_path 图像路径 --visualize
```

特征可视化效果图会保存在feat文件中



开发不易，**训练代码**部分有偿提供~ 

Wechat:y24065939s

CSDN:https://blog.csdn.net/z240626191s/article/details/137942794?fromshare=blogdetail
