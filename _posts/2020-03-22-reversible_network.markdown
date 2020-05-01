---
layout: draft
title: "Reversible network"
permalink: /reversible_network
date: 2020-03-22 14:49:0 +0000
comments: False
share: False
categories: ml
---

**related materials**

check [Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/pdf/1912.02762.pdf)

[Normalizing Flows: An Introduction and Review of Current Methods](https://arxiv.org/pdf/1908.09257.pdf)

[Invertible Models and Normalizing Flows: a retrospective (ICLR 2020 keynote slides)](https://docs.google.com/presentation/d/15RMCzCRwuKKv6fIwvGjwig2WnnP_5yzQGzcpJbq7zws/edit#slide=id.g8428c68825_0_0)


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

**[Coupling Flow]()**

**[NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION,ICLRW15](https://arxiv.org/abs/1410.8516)**


**[Density estimation using Real NVP,ICLR17](https://arxiv.org/abs/1605.08803)**


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
 


