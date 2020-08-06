---
layout: draft
title: "Cloud Point"
permalink: /cloud_point
date: 2020-03-31 14:49:0 +0000
comments: False
share: False
categories: cv
---

[https://github.com/nicolas-chaulet/torch-points3d](https://github.com/nicolas-chaulet/torch-points3d)


## Unsupversed

**[Total Denoising: Unsupervised Learning of 3D Point Cloud Cleaning,ICCV19](https://arxiv.org/pdf/1904.07615.pdf)**

**Task Unsupervised denoising:** Our results demonstrate unsupervised denoising performance similar to that of supervised learning with
clean data when given enough training examples - whereby
we do not need any pairs of noisy and clean training data.

**[Unsupervised Multi-Task Feature Learning on Point Clouds,ICCV19](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hassani_Unsupervised_Multi-Task_Feature_Learning_on_Point_Clouds_ICCV_2019_paper.pdf)**

**[Self-supervised Modal and View Invariant Feature Learning,Arxiv2005](https://arxiv.org/pdf/2005.14169.pdf)**



**[PointContrast: Unsupervised Pre-training for 3D Point Cloud Understanding,ECCV20,spotlight](https://arxiv.org/pdf/2007.10985.pdf)**



**[Mapping in a Cycle: Sinkhorn Regularized Unsupervised Learning for Point Cloud Shapes,ECCV20](https://arxiv.org/pdf/2007.09594.pdf)**


**[Self-Supervised Learning of Point Clouds via Orientation Estimation,Arxiv2008](https://arxiv.org/pdf/2008.00305.pdf)**


## Point Cloud Augmentation

**[PointAugment: an Auto-Augmentation Framework
for Point Cloud Classification,CVPR20,oral](https://arxiv.org/pdf/2002.10876.pdf)**

## Point cloud classification

**[PointNet++]()**

## Point Sampling

[Learning to Sample,CVPR19](https://arxiv.org/pdf/1812.01659.pdf)


![](/imgs/learn-to-sample.png)


The input to S-NET is a set of n 3D coordinates, namely
points, representing a 3D shape. The output of S-NET is
k generated points. 


Detail structure of S-Net can be seen in A.1, check [code](https://github.com/itailang/SampleNet/blob/master/registration/src/samplenet.py#L82).

G maybe is not a subset P, do matching process by KNN.


construct a sampling regularization loss,composed out of three terms:


$$
L_{f}(G,P) = \frac{1}{|G|} \sum_{g \in G} min_{p \in P} ||p - g||^{2}_{2} 
$$

$$
L_{b}(G,P) = \frac{1}{|P|} \sum_{p \in P} min_{G \in G} ||p - g||^{2}_{2} 
$$


$$
L_{m}(G,P) = max_{g \in G} min_{p \in P} ||g-p||^{2}_{2}
$$



$$L_{f}(G,P) + L_{b}(G,P)$$ is a typical Chamfer Distance(CD).


**[SampleNet: Differentiable Point Cloud Sampling,CVPR20](https://arxiv.org/pdf/1912.03663.pdf)**


extend previous works by introducing a differentiable relaxation to the matching step, i.e., nearest neighbor selection, during training.

Temperature in softmax is learnable, counter-intuitive.





