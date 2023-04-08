# CV_HW1：构建两层神经网络分类器

刘佳欣-19307130299

数据集：MINIST；

作业要求：至少包含以下三个代码文件/部分
  1. 训练：激活函数，反向传播，loss以及梯度的计算，学习率下降策略，L2正则化，优化器SGD，保存模型
  2. 参数查找：学习率，隐藏层大小，正则化强度
  3. 测试：导入模型，用经过参数查找后的模型进行测试，输出分类精度

注：不可使用pytorch，tensorflow等python package，可以使用numpy；

## 文件说明

- `HW1_report_19307130299.pdf`: 实验报告
- `train.py` : 训练神经网络
- `test.py`: 测试模型
- `utils`: 自定义辅助函数，以及用于可视化参数
- `model.py`： 模型基本结构及梯度更新
- `try_coefficient`: 在不同超参数列表中选取最优超参数
- `best_model.mat`：模型
- `Error_best.jpg`,`Loss_best.jpg`: 训练过程中的Error和Loss曲线
- `X_test.npy`,`X_train.npy`,`y_test.npy`,`y_train.npy`：MNIST数据集

## 代码运行步骤

### 训练步骤：
```shell
python train.py
```
默认参数为最优参数，也可指定参数，如
```shell
python train.py --lr 0.1 --iter 300 --regular 0.2 --regular 0.01 --lr_decay 0.1
```
训练过程中会自动绘制Loss和Error图像并保存，训练完成后会保存最优模型`para.mat`，为了不覆盖现有模型和图片，与文件中现有命名不同

### 测试步骤：
```shell
python test.py
```
可以指定模型，无需后缀和单引号
```shell
python test.py --para_path best_model
```
测试会输出模型在测试集上的准确率

### 最优超参数查找
```shell
python try_coefficient.py --lr 1e-2 1e-3 1e-4 --iter 300 --layer 50 100 300 --regular 1e-2 1e-3 1e-4 1e-5
```
可以在给定范围中返回最优参数