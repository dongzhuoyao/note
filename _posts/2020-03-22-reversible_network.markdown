---
layout: draft
title: "Reversible network"
permalink: /reversible_network
date: 2020-03-22 14:49:0 +0000
comments: False
share: False
categories: ml
---

check [Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/pdf/1912.02762.pdf)

[Normalizing Flows: An Introduction and Review of Current Methods](https://arxiv.org/pdf/1908.09257.pdf)


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


**[RealNVP]()**

**[Glow](https://arxiv.org/pdf/1807.03039.pdf)**


![](/imgs/glow.png)

ActNorm is similar to BN, without mean and standard deviation. only learn the scale and bias with size $$C\times 1\times 1$$

An additive coupling layer proposed before is a special case with s = 1 and a log-determinant of
0 in affine coupling layers.

**[Inverse Autoregressive Flow (IAF)](https://papers.nips.cc/paper/6581-improved-variational-inference-with-inverse-autoregressive-flow)**


**[RevNets]()**

**[iRevNets]()**

**[A Disentangling Invertible Interpretation Network for Explaining Latent Representations,Arxiv2004](https://arxiv.org/pdf/2004.13166.pdf)**
 


