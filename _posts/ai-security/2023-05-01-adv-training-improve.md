---
title: "通过对抗训练来提高图像识别的精度"
date: 2023-05-01 13:38:00 +0800
math: true
categories: [对抗样本]
tags: [对抗训练, 图像识别] 
---


### 介绍

这次介绍一篇CVPR2020的工作，**Adversarial Examples Improve Image Recognition**，该工作主要揭示了对抗样本对图像分类的促进作用。

> 关于对抗样本，可以查看我的这篇[文章](https://mezereonxp.fun/posts/adv-attacks)



### 对抗训练的影响

对抗样本一直以来大家对其印象都不好，使分类器出错，难以进行防御，白盒攻击下的防御大多数难以真正完全防御住。

先来看看目前相对比较有效的防御，即对抗训练，如下图所示：

![image-20210408200506478](https://mezereon-upic.oss-cn-shanghai.aliyuncs.com/uPic/image-20210408200506478.png)

灰色代表着对抗训练，橘色代表对抗训练加上参数调优（Fine-tuning，其实就是再重新训练少数次）。

可以看到，对抗训练（简记为AT）之后会带来一定程度的正确率下降，但是，经过重训练之后，正确率竟可能恢复甚至比原有的精度高。

这种现象，可以理解为是数据扩充之后的影响，对抗样本可以理解为原有样本的一个相邻样本。

只不过，由于对抗样本和正常样本的分布存在差异，因此，正确率一开始会下降。

那么如何利用这种分布不一致的特殊的“扩充数据”呢？

### 加一个BN试一试

在先前的对抗训练之中，如下图所示：

![image-20210408201322630](https://mezereon-upic.oss-cn-shanghai.aliyuncs.com/uPic/image-20210408201322630.png)

由于对抗样本和干净样本的分布不同，但是却过的同一个BN进行训练，这会使得BN学习到的超参数趋近于二者的联合分布。

直觉地，我们直接加一个额外的BN，专门给对抗样本训练用

![image-20210408201516402](https://mezereon-upic.oss-cn-shanghai.aliyuncs.com/uPic/image-20210408201516402.png)

我们生成对抗样本的时候也使用这个额外的BN层，训练的时候也一样。

在使用的时候，我们只需要将这个额外的BN层撤去即可。



### 实验评估

文章在EfficientNet的不同规模的网络上进行测试，如下图所示

![image-20210408201840129](https://mezereon-upic.oss-cn-shanghai.aliyuncs.com/uPic/image-20210408201840129.png)

在ImageNet上基本上都取得了一定的提升，只不过提升程度有限。



**同时，该方法可以增强网络的泛化能力，如下表所示**

![image-20210408202104515](https://mezereon-upic.oss-cn-shanghai.aliyuncs.com/uPic/image-20210408202104515.png)

文章在其他数据集上进行了泛化能力的测试，结果均表明所采用方法能够有效增加模型的泛化能力。



**和直接加对抗训练的比较**

![image-20210408202249608](https://mezereon-upic.oss-cn-shanghai.aliyuncs.com/uPic/image-20210408202249608.png)

可以看到，相较于直接的对抗训练，这种方法一定程度上减弱了分布差异带来的影响



**和传统的数据增强方案的比较**

![image-20210408210855777](https://mezereon-upic.oss-cn-shanghai.aliyuncs.com/uPic/image-20210408210855777.png)