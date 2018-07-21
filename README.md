# attribute recognition

#### 环境
`pytorch`, #版本0.3.0or0.3.1

`tensorboard_logger`, #log信息

`pretrainedmodels`, #预训练模型库

`argparse`,
`PIL`

#### 1.目录
文件目录：
```
fashionai_demo
├── train.sh
├── data
│   ├── base
│   ├── rank
│   └── web
├── prepare_data.py
├── README.md
└── ...
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
