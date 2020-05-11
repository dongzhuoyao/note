---
layout: draft
title: "Metric Learning"
permalink: /metric_learning
date: 2020-03-31 14:49:0 +0000
comments: False
share: False
categories: cv
---
<!--

https://www.zhihu.com/question/382802283/answer/1118867880

https://zhuanlan.zhihu.com/p/136522363

-->

**[Cross-Batch Memory for Embedding Learning,CVPR20,oral](https://arxiv.org/pdf/1912.06798.pdf)**

<!--
https://zhuanlan.zhihu.com/p/136522363
-->

**[NormFace: L2 Hypersphere Embedding for Face Verification,ACM17](https://arxiv.org/pdf/1704.06369.pdf)**

<!--
https://www.zhihu.com/question/67589242
-->

**[SoftTriple Loss: Deep Metric Learning Without Triplet Sampling,ICCV19](https://arxiv.org/pdf/1909.05235.pdf)**


**[Hyperbolic Image Embeddings](https://arxiv.org/pdf/1904.02239.pdf)**

Useful for image classification, retrieval, few-shot learning

**EMD(Earth Moving Distance)**

![](/imgs/emd.png)

pic from [this paper](https://arxiv.org/pdf/2003.06777.pdf)

You need to know the cost per unit c, and the supply amount s and demand amount d in advance.

**EMD in cloud point** starts from [this paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Fan_A_Point_Set_CVPR_2017_paper.pdf)

What is c,s,d in Cloud Point? it seems that supply and demand is not a must in this optimal transport problem formulation, check powerpoint [here](http://imagine.enpc.fr/~langloip/data/OptimalTransport.pdf).

It's quite similart to chamfer distance. While chamfer distance is not a bijection, EMD is a bijection.






