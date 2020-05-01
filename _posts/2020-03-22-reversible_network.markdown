---
layout: draft
title: "Reversible network"
permalink: /reversible_network
date: 2020-03-22 14:49:0 +0000
comments: False
share: False
categories: ml
---

## related materials

check [Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/pdf/1912.02762.pdf)

[Normalizing Flows: An Introduction and Review of Current Methods](https://arxiv.org/pdf/1908.09257.pdf)

[Invertible Models and Normalizing Flows: a retrospective (ICLR 2020 keynote slides)](https://docs.google.com/presentation/d/15RMCzCRwuKKv6fIwvGjwig2WnnP_5yzQGzcpJbq7zws/edit#slide=id.g8428c68825_0_0)


## Pre history: distribution estimation

[NADE:Neural Autoregressive Distribution Estimation,JMLR2000](https://arxiv.org/pdf/1605.02226.pdf)

## Recent Advance

**[Planar and Radial Flows]()**

Planar Flow:
$$
g(x) = x + u h(w^{T}x +b)
$$

An extension based on Planar flow is Sylvester flow:
$$
g(x) = x + U h(W^{T}x +b)
$$
where U and W are DxM matrices

Radial Flows:

$$
g(x) = x + \frac{\beta}{\alpha + ||x-x_{0}||}(x-x_{0})
$$


**[NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION,ICLRW15](https://arxiv.org/abs/1410.8516)**

core idea of coupling layer(actually also proposed a genneral coupling layer, while they use additive coupling layer for simplicity.):

$$
y_{1} =x_{1}\\
y_{2} = x_{2} + m(x_{1})
$$

m can be as complex as you need, I like this idea, why the fucking ICLR reject it? Also this paper is honest compared with common papers. show the simple intuition in the very beginning.

> Examining the Jacobian, we observe that at least
three coupling layers are necessary to allow all dimensions to influence one another. We generally use four.

Prior distribution can be gaussian distribution or logistic distribution. Their prior distribution can be explicitly expressed in session3.4(EXCERCISE)

Difference between VAE: Like the variational auto-encoders, the NICE model uses an encoder to avoid the difficulties of inference, but its encoding is deterministic. The log-likelihood is tractable and the training procedure does not require any sampling (apart from dequantizing the data).

SCALING intuition: As each additive coupling layers has unit Jacobian determinant (i.e. is volume preserving), their composition will necessarily have unit Jacobian determinant too.(TODO)This allows the learner to give more weight (i.e. model more variation) on some dimensions and less in others. similar to attention mechanism recently. 

The INPAINTING application is interesting, a super simple projected gradient ascent is applied based on the pre-trained combination probability between H and O. 

The change of variable formula for probability density functions is prominently used, check related works in this paper.

The NICE criterion is very similar to the criterion of the variational auto-encoder. More specifically,
as the transformation and its inverse can be seen as a perfect auto-encoder pair,... check related work.TODO

**[Density estimation using Real NVP,ICLR17](https://arxiv.org/abs/1605.08803)**


Contributions: affine coupling layer, masked convolution, multi-scale architecture(squeeze out), introduce moving-average batch normalization into this topic.

Training a normalization flow does not in theory requires a discriminator network as in GANs, or approximate inference as in variational autoencoders. If the function is bijective, it can be trained through maximum likelihood using the change of variable formula. This formula has been discussed in several papers including the maximum likelihood formulation of independent components analysis (ICA) [4, 28], gaussianization [14, 11] and deep density models [5, 50, 17, 3]. 



**[Glow](https://arxiv.org/pdf/1807.03039.pdf)**


![](/imgs/glow.png)

ActNorm is similar to BN, without mean and standard deviation. only learn the scale and bias with size $$C\times 1\times 1$$

An additive coupling layer proposed before is a special case with s = 1 and a log-determinant of
0 in affine coupling layers.



**[Inverse Autoregressive Flow (IAF)](https://papers.nips.cc/paper/6581-improved-variational-inference-with-inverse-autoregressive-flow)**


**[RevNets]()**

**[iRevNets]()**

**[Do Deep Generative Models Know What They Don't Know?,ICLR19](https://arxiv.org/pdf/1810.09136.pdf)**

**[FFJORD: Free-Form Continuous Dynamics for Scalable Reversible Generative Models,ICLR19](https://openreview.net/forum?id=rJxgknCcK7)**

**[PointFlow: 3D Point Cloud Generation with Continuous Normalizing Flows,ICCV19](https://arxiv.org/abs/1906.12320)**

**[C-Flow: Conditional Generative Flow Models for Images and 3D Point Clouds,ICCV19]()**

**[Hybrid Models with Deep and Invertible Features,ICML19](https://arxiv.org/pdf/1902.02767.pdf)**


**[Invert to Learn to Invert Patrick,NIPS19](https://arxiv.org/abs/1911.10914)**

**[A Disentangling Invertible Interpretation Network for Explaining Latent Representations,Arxiv2004](https://arxiv.org/pdf/2004.13166.pdf)**
 


