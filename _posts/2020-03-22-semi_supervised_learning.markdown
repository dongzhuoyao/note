---
layout: draft
title: "Semi-supervised learning"
permalink: /semi_supervised_learning
date: 2020-03-22 14:49:0 +0000
comments: False
share: False
categories: ml
---

## Basic solution

#### Normalization flow-based

Train:

$$
L(\theta) = \sum_{(x_{i},y_{i}) \in labelled} log p_{\theta} (x_{i}, y_{i}) + \sum_{x_{j} \in unlabelled} log p_{\theta}(x_{j})
$$

$$p_{\theta} (x_{i}, y_{i})$$ can be solved by(1). _Hybrid models_[^hybridmodel] or (2). further decomposed as$$
p_{\theta} (x_{i}, y_{i}) = p_{\theta}(y_{i}) * p_{\theta}(x_{i}|y_{i})
$$

(1). Hybrid model decompose
$$
p_{\theta} (x_{i}, y_{i})
$$  to  $$
p_{\theta}(x_{i})*p_{\theta}(y_{i}|x_{i})
$$

 This paper  use Generalized Linear Models(GLM) to model 
 $$
 p(y_{i}|x_{i})
 $$, P(x) is modeled by Normalization Flow.

(2). $$
p(x_{i})
$$ is modeled by normalization flow, $$
p(x_{i}|y_{i})
$$ is modeled by conditional normalization flow(novelty lies in).


Testing:
$$p(y|x) = p(x,y)/p(x)$$

#### Semi-supervised image classification

**[TEMPORAL ENSEMBLING FOR SEMI-SUPERVISED
LEARNING,ICLR17](https://arxiv.org/pdf/1610.02242.pdf)**

![](/imgs/temporal-ensembling.png)


$$z_{i}$$ is $$N \times C$$, will moving averaged to $$\tilde{z_{i}}$$, check alg 1 in the paper.



**[Mean Teacher,Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results](NIPS17)**

Motivated by Temporal Emsembling.



Three different noises are considered: The model architecture is a 13-layer convolutional neural network (ConvNet) with three types of noise: random translations and horizontal flips of the input images, Gaussian noise on the input layer, and dropout applied within the network.



**CutMix**

## Semi-supervised semantic segmentation

**Consistency regularization**

Consistency regularization (Sajjadi et al., 2016b; Laine & Aila, 2017; Miyato et al., 2017; Oliver
et al., 2018) describes a class of semi-supervised learning algorithms that have yielded state-ofthe-art results in semi-supervised classification, while being conceptually simple and often easy to
implement. The key idea is to encourage the network to give consistent predictions for unlabeled
inputs that are perturbed in various ways.



**[Adversarial Learning for Semi-Supervised Semantic Segmentation,BMVC18](https://arxiv.org/pdf/1802.07934.pdf)**

![](/imgs/adv-semi-seg.png)

Pay more attention to $$L_{semi}$$ loss, by thresholding the output of discriminator network to construct a psydo label.


**[CowMix, Semi-supervised semantic segmentation needs strong, high-dimensional perturbations ](https://openreview.net/forum?id=B1eBoJStwr)**

> The use of a rectangular mask restricts the dimensionality of the perturbations that CutOut and
CutMix can produce. Intuitively, a more complex mask that has more degrees of freedom should
provide better exploration of the plausible input space. We propose combining the semantic CutOut
and CutMix regularizers introduced above with a novel mask generation method, giving rise to two
regularization methods that we dub CowOut and CowMix due to the Friesian cow -like texture of
the masks.



#### Footnotes
* footnotes will be placed here. This line is necessary
{:footnotes}

[^hybridmodel]: Hybrid Models with Deep and Invertible Features, ICML19.




