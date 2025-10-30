# 深度学习装甲板分类

我本身是搞这个方向的，这个题对我没难度（）

框架采用ResNet34，训练100个回合，采用标准AlexNet对ImageNet数据增强技术

最高学习率0.01，Momentum设置为0.9，采用WarmUp热启动，

一共训练了4个小时训练过程已写入TensorFlow日志文件，用网页打开即可查看

记录了Acc，Train_Loss和Test_Loss，测试集最高Acc达到99.58%

想要测试我的代码得先解压数据集（
