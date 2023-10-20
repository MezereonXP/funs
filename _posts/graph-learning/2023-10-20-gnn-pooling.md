---
title: "Graph Pooling 笔记"
date: 2023-10-20 21:37:00 +0800
math: true
categories: [图学习]
tags: [图池化]
---

# Graph Pooling 笔记

> Pooling 是一种用于图表征提取的技术，通常用在图分类上面。

### 一些记号

我们记一个带有 $N$ 个节点的属性图 (attributed graph) 为 $\mathcal{G} = (\mathcal X, \mathcal E)$

其中 $\mathcal X =\{(i,x_i)\}_{i=1:N}$ 是节点集，$x_i$ 是第 $i$ 个节点的属性向量

$\mathcal E = \{((i,j), e_{ij})\}_{i,j\in 1:N}$ 是边集，其中 $e_{ij}$ 是边的属性向量

我们记这个图的邻接矩阵为 $A \in \{0,1\}^{N\times N}$

借助论文“Understanding Pooling in Graph Neural Networks” 我们使用其中的 **SRC** 来对Pooling方法进行总结。

### Select, Reduce, Connect

对于Pooling，我们可以理解成一个图到图的映射，即：$\mathcal G \rightarrow \mathcal G' = (\mathcal X', \mathcal E')$

![image-20220217213747408](https://mezereon-upic.oss-cn-shanghai.aliyuncs.com/uPic/image-20220217213747408.png)

如上图所示，Select函数会将节点划分成多个节点簇，这些节点簇可以被认为是一个超节点

Reduce函数会将一个超节点（可能包含一个或多个节点）映射到一个属性向量，该属性向量对应Pooling后图的超节点

Connect函数会计算出超节点的边集

![image-20220217214217638](https://mezereon-upic.oss-cn-shanghai.aliyuncs.com/uPic/image-20220217214217638.png)

在Pooling操作之后，我们将一个N节点的图映射到一个K节点的图

按照这种方法，我们可以给出一个表格，将目前的一些Pooling方法，利用SRC的方式进行总结

![image-20220218140909561](https://mezereon-upic.oss-cn-shanghai.aliyuncs.com/uPic/image-20220218140909561.png)

这里以 **DiffPool** 为例，说明一下SRC三个部分：

首先，假设我们有一个N个节点的图，其中节点向量记作 $X\in \mathbb R^{N\times d}$，每个节点向量的维度是 $d$

Select函数会输出一个 $N\times N' (N <N')$ 的映射矩阵$S$, 也就是将$N$个点映射成$N’$个点 

这里面使用了一个GNN来对矩阵$S$ 进行学习

Reduce函数为映射矩阵S乘上一个GNN之后的矩阵，也就是$N'\times N$ 的矩阵乘上 $N\times d'$

输出为 $N'\times d$ 的一个向量矩阵，代表Pooling一次之后的这些超节点的节点向量

Connect函数输出邻接矩阵$A'\in \{0,1\}^{N'\times N'}$

