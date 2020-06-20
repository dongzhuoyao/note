---
layout: draft
title: "deep learning stones"
permalink: /dl_stones
date: 2020-03-22 14:49:0 +0000
comments: False
share: False
categories: cv
---


## Augmentation

**[AutoAugment: Learning Augmentation Strategies from Data,CVPR19](https://arxiv.org/abs/1805.09501)**



## Learning rate

**[LARS:Large Batch Training of Convolutional Networks,Arxiv1708](https://arxiv.org/abs/1708.03888)**

##  Normalization 

check related work in [ICLR20](https://arxiv.org/pdf/2001.06838.pdf).

Input tensor size $$NxCxHxW$$


![](/imgs/normalization.png)

**Batch Normalization**

mean, std tensor size is $$C \times 1$$

mainly used in segmentation, pixelwise instance normalization(each instance is a pixel rather than a image)

**Batch Renormalization**

BR  introduces two extra parameters (r and d) that constrain the estimated mean and variance of
BN. Their values are controlled by $$r_{max}$$ and $$d_{max}$$.

**[Layer Normalization](https://arxiv.org/abs/1607.06450)**

mean, std tensor size is $$N \times 1$$

Empirically, there is no need for affine tranformation(they name it as "bias ang gain parameter") in Layer Normalization, check [Understanding and Improving Layer Normalization](https://papers.nips.cc/paper/8689-understanding-and-improving-layer-normalization.pdf). Therefore, there is no extra trainable parameter in Layer Normalization.



Another difference between BN mentioned in [this unsupervised-learning paper](https://openreview.net/pdf?id=rkxoh24FPH)

> LayerNorm avoids the possibility of information leakage within mini-batches that can be induced through
batch normalization, potentially leading to poor performance

![](/imgs/invariance_normalization.png)

Why LN is invariant when weight matrix re-centering?



**Instance Normalization**

mean, std tensor size is $$N \times C$$

**Group Normalization**

mean, std tensor size is $$N \times C/K$$

GN becomes LN if we set the group number as G = 1.

GN becomes IN  if we set the group number as G = C (i.e., one channel per group).


**[Rethinking Normalization and Elimination Singularity in Neural Networks,Arxiv1911](https://arxiv.org/pdf/1911.09738.pdf)**

equip channel-based normalization(IN,LN,GN) with batch information,
and the result is Batch-Channel Normalization.

![](/imgs/bcn.png)

**[Understanding the Disharmony between Weight Normalization Family and Weight Decay:−shifted L2 Regularizer,Arxiv](https://arxiv.org/pdf/1911.05920.pdf)**



**[Weight normalization: A simple reparameterization to accelerate training of deep neural networks,NIPS16](https://arxiv.org/pdf/1602.07868.pdf)**


Weight normalization can thus be viewed as a cheaper and less noisy approximation to batch
normalization.

Given a activation:
$$
y = \phi(w \dot x +b)
$$

reparameterize each weight vector w in terms of a parameter vector v and a scalar parameter g and to perform stochastic gradient descent with respect to those parameters instead.

$$
w =\frac{g}{||v||} v
$$


**Data-Dependent Initialization of Parameters matters**: this method ensures that all features initially have zero mean and unit variance before application of the nonlinearity. 

similar intuition occurs in STN: STN network, try to learning translation matrax rather than directly learning the translated result. Deep learning is a block box, decompose the problem towards a better form will help optimization.

similar thought is also applied in normalization flow: [ActNorm in Glow](https://github.com/rosinality/glow-pytorch/blob/master/model.py#L11)



**[Weight Standardization,Arxiv1903](https://arxiv.org/pdf/1903.10520.pdf)**

[code](https://github.com/joe-siyuan-qiao/WeightStandardization),[zhihu](https://www.zhihu.com/question/317725299)

![](/imgs/weight-standardization.png)

for im2col matrix [NWH,KKC], the mean, variance size is [KKC].

- Note that we do not have any affine transformation on Wˆ . This is because we assume that normalization layers such as BN or GN will normalize this convolutional layer again, and having affine transformation will harm training as we will show in the experiments.
  


**[Normalized Convolutional Neural Network,Arxiv2005](https://arxiv.org/pdf/2005.05274.pdf)**

[code](https://github.com/kimdongsuk1/NormalizedCNN/blob/master/NCNN.py)

for im2col matrix [NWH,KKC], the mean, variance size is [NWH]


- can be seen as dual version of weight Standardization. In weight Standardization, for im2col matrix [NWH,KKC], the mean, variance size is [KKC].
- For all models, training batch is set to 2.
- In batch independent normalization methods, Positional Normalization(PN)[10] is similar with out method because if a kernel size is 1x1, the process is same . But kernel size is not always 1x1, our method is more adaptive to direct concerned slice-inputs as NC standardizes im2col matrix.
- When we conduct experiment using several optimizer method, we can’t obtain good results on other gradient adpative optimizer method such as Adam[15],RMSProp. Therefore NC need a new adpative optimizing methods.



**[Four Things Everyone Should Know to Improve Batch Normalization,ICLR20](https://openreview.net/forum?id=HJx8HANFDH)**


**[Towards Stabilizing Batch Statistics in Backward Propagation of Batch Normalization,ICLR20](https://openreview.net/forum?id=SkgGjRVKDS)**

motivated by BN back propogation.



[ICLR review](https://openreview.net/forum?id=SkgGjRVKDS)

> The paper extends recently proposed BatchRenormalization (BRN) technique which uses exponential moving average (EMA) statistics in forward and backward passes of BatchNorm (BN) instead of vanilla batch statistics. Motivation of the work is to stabilize training neural networks on small batch size setup. Authors propose to replace EMA in backward pass by simple moving average (SMA) and show that under some assumptions such replacement reduces variance. Also they consider slightly different way of normalization without centralizing features X, but centralizing convolutional kernels according to Qiao et al. (2019).

**[Riemannian approach to batch normalization,NIPS17](https://arxiv.org/pdf/1709.09603.pdf)**

**[Improving training of deep neural networks via Singular Value Bounding,CVPR17](https://arxiv.org/pdf/1611.06013.pdf)**

**[All You Need is Beyond a Good Init: Exploring Better Solution for Training Extremely Deep Convolutional Neural Networks with Orthonormality and Modulation,CVPR17](https://arxiv.org/pdf/1703.01827.pdf)**

**[Training BatchNorm and Only BatchNorm: On the Expressive Power of Random Features in CNNs,Arxiv2003](https://arxiv.org/pdf/2003.00152.pdf)**

Check the expressive power of coefficient annd bias in BN. 

> To study this
question, we investigate the performance achieved
when training only these parameters and freezing
all others at their random initializations. We find
that doing so leads to surprisingly high performance. For example, a sufficiently deep ResNet
reaches 83% accuracy on CIFAR-10 in this configuration. 




## Upsampling 


**[Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network,CVPR16](https://arxiv.org/abs/1609.05158)**

![](/imgs/subpixelconv.png)

Also used in i-revnet.


[Understanding Convolution for Semantic Segmentation,WACV18](https://arxiv.org/abs/1702.08502)


Similar to aforementioned sub-pixel convolution.

<!-- [Split-Merge Pooling](https://arxiv.org/pdf/2006.07742.pdf) -->







## Pooling

#### Bilinear Pooling


**[Bilinear CNN](http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf)**

check code for understading: [https://github.com/HaoMood/bilinear-cnn/blob/master/src/bilinear_cnn_all.py#L71](https://github.com/HaoMood/bilinear-cnn/blob/master/src/bilinear_cnn_all.py#L71)


**[Factorized Bilinear Models for Image Recognition](http://openaccess.thecvf.com/content_ICCV_2017/papers/Li_Factorized_Bilinear_Models_ICCV_2017_paper.pdf)**

Although the bilinear pooling is capable of capturing pairwise interactions, it also introduces a quadratic
number of parameters in weight matrices $$W^{R}_{j}$$ leading to
huge computational cost and the risk of overfitting.

$$
y = b + w^{T}x +x^{T}F^{T}Fx
$$

where $$F \in \mathbb{R}^{k \times n}$$

The key idea of our DropFactor is to randomly drop the bilinear paths corresponding to k factors during the training.

performs nearly the same with bilinear pooling in fine-grained classification task. therefore the author mainly focus on the experiment of traditional classifcation task. check last subsession.



**[MPN-COV](https://arxiv.org/pdf/1703.08050.pdf)**




#### [Strip Pooling,CVPR20](https://arxiv.org/pdf/2003.13328.pdf)

Check Fig2, the spatial dimension doesn't change, why name it ``pooling''?

Unlike the two-dimensional average pooling, the proposed strip pooling averages all the feature values in a row or a column. Given the horizontal and vertical strip pooling layers, it is easy to build long-range dependencies between regions distributed discretely and encode regions with the banded shape, thanks to the long and narrow kernel shape.

check implementation [here](https://github.com/Andrew-Qibin/SPNet/blob/master/models/customize.py).



#### Spatial Pyramid Pooling

By adopting a set of parallel pooling operations with a unique kernel size at each pyramid level, the network is able to capture largerange context. It has been shown promising on several scene parsing benchmarks.its ability to exploit contextual information is limited since only square kernel shapes are applied. Moreover, the spatial pyramid pooling is only modularized on top of the backbone network thus rendering it is not flexible or directly applicable in the network building block for feature learning. 


## Conv

**[Dynamic Region-Aware Convolution,Arxiv2003](https://arxiv.org/pdf/2003.12243.pdf)**

[https://zhuanlan.zhihu.com/p/136998353](https://zhuanlan.zhihu.com/p/136998353)


**im2col tricks**


[https://zhuanlan.zhihu.com/p/63974249](https://zhuanlan.zhihu.com/p/63974249)


Im2col size is [N(H-k+1)(W-k+1), kxkxc], convolution kernel size is [kxkxc, d].

The reason of naming 'im2col', for the size of im2col matrix, each column is N(H-k+1)(W-k+1) sized, it represents the whole image. There is also im2row, the size is [kxkxc, N(H-k+1)(W-k+1)].

When will im2row be used?

Why caffee is BCWH and torch is BWHC?

[Row- and column-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order)

[Why GEMM is at the heart of deep learning](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/)

[Low-memory GEMM-based convolution algorithms for deep neural networks](https://arxiv.org/pdf/1709.03395.pdf)

[efficient CPU version, as_strided](https://zhuanlan.zhihu.com/p/64933417)

[The Indirect Convolution Algorithm](https://arxiv.org/pdf/1907.02129.pdf)
[zhihu discussion](https://www.zhihu.com/question/336558535)


## Back-propogation

**[ReSprop: Reuse Sparsified Backpropagation,CVPR20oral](http://openaccess.thecvf.com/content_CVPR_2020/papers/Goli_ReSprop_Reuse_Sparsified_Backpropagation_CVPR_2020_paper.pdf)**


- The analysis of Figure 2 is interesting, by using "o hyperdimensional computing
theory [25], two independent isotropic vectors picked randomly from a high dimensional space d, are approximately orthogonal." and "empirial 37 degree".
- learning rate warm up is important(we are using 5 to 8 epochs of whole training (90-200
epochs) for the warm up phase), especially when larger RS(reuse strategy).  check Table3.
- Impact of batch size: ReSprop and W-ReSprop algorithms achieve higher accuracy for larger batch sizes.
- also can handle distributed-GPU training, if your batch size is larger as 128, check Table 7.
- Adaptive thresholding: halve or double by some strategy. **chose the initialization value of 10−7
for all the layers in all the experiments, based on the output gradient’s distribution on the ResNet-18, 34 and 50 on CIFAR datasets**
- citation: Amdahl's law to theoretically analysis


**[meProp: Sparsified Back Propagation for Accelerated Deep Learning,ICML17](http://proceedings.mlr.press/v70/sun17c/sun17c.pdf)**

with the background of Resprop, it's easy to understand: meProp only select top-k(by magnitude) by setting others as zero. Check intersting Table 1 in ReSprop(CVPR20).

**[Dynamic Sparse Graph for Efficient Deep Learning,ICLR19](https://openreview.net/forum?id=H1goBoR9F7)**

too complex



## Optimizer

**[AdaX: Adaptive Gradient Descent with Exponential Long Term Memory,Arxiv2004](https://arxiv.org/pdf/2004.09740.pdf)**

**[AdaBound: Adaptive Gradient Methods with Dynamic Bound of Learning Rate,ICLR19](https://openreview.net/forum?id=Bkg3g2R9FX)**



