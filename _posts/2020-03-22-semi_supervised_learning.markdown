---
layout: draft
title: "Semi-supervised learning"
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

#### Semi-supervised semantic segmentation

## Mean Teacher

#### Footnotes
* footnotes will be placed here. This line is necessary
{:footnotes}

[^hybridmodel]: Hybrid Models with Deep and Invertible Features, ICML19.




