---
layout: draft
title: "Cloud Point"
permalink: /cloud_point
date: 2020-03-31 14:49:0 +0000
comments: False
share: False
categories: cv
---

## Point cloud classification



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





