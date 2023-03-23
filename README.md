# CV_HW1：构建两层神经网络分类器

刘佳欣-19307130299

数据集：MINIST；

作业要求：至少包含以下三个代码文件/部分
  1. 训练：激活函数，反向传播，loss以及梯度的计算，学习率下降策略，L2正则化，优化器SGD，保存模型
  2. 参数查找：学习率，隐藏层大小，正则化强度
  3. 测试：导入模型，用经过参数查找后的模型进行测试，输出分类精度

注：不可使用pytorch，tensorflow等python package，可以使用numpy；

### 训练步骤：
```shell
python train.py
```
可指定参数，如
```shell
python train.py --lr 0.1 --regular 0.2
```

### 测试步骤：
```shell
python test.py
```

### 文件说明

- `report.pdf`: 实验报告
- `train.py` : 对传入参数训练神经网络
- `test.py`: 测试模型
- `utils`: 自定义辅助函数
