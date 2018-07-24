# attribute recognition
这个比赛当时和杭电的陶星一起组队的，然后也加了一些其他人，但是主要是我和陶星在做。一段很棒的经历，水平提升不少。最终排名：7/2950
#### 环境
`pytorch`, #版本0.3.0or0.3.1

`tensorboard_logger`, #log信息

`pretrainedmodels`, #预训练模型库

`argparse`,
`PIL`

#### 1.目录
文件目录：
```

```
#### 2.准备数据
将`训练集`，`测试集`，`热身数据`分别放入`base`,`rank`,`web`

运行prepare_data.py进行数据划分
#### 3.训练
`main.py` #`--arch`为模型名称，不同预训练模型在最后一层不一样，需要调试修改

`train.sh` #训练脚本
#### 4.生成csv
`evaluate.py` #`arc_lists`为待测试模型
#### 5.结果平均
`average.ipynb`
