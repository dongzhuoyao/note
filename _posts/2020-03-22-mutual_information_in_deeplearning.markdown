---
layout: draft
title: "Mutual Information in Deep Learning"
date: 2020-03-22 14:49:0 +0000
comments: False
share: False
categories: ml
---

# Self-supervised learning

CPC: Representation Learning with
Contrastive Predictive Coding

#### Basic idea

latent representations $$z_{t}=g_{enc}(x_{ts})$$. An autoregressive model $$g_{ar}$$ summarizes all z<=t  in the latent space and produces a context latent representation $$c_{t}=g_{ar}(z\le t)$$.

Mutual information between original signal x and c:
$$
I(x;c) = \sum_{x,c} p(x,c) log \frac{p(x|c)}{p(x)}
$$


predict future observations as $$\frac{p(x_{t+k}|c_{t})}{p(x_{t+k})}$$ 
rather than 
$$p_{k}(x_{t+k}|c_{t})$$

In reality, a simple log-bilinear model $$
f_{k}(x_{t+k},c_{t})=exp(z^{T}_{t+k}W_{k}c_{t})
$$ is used.

Given X=$$x_{1},...,x_{N}$$ of N random samples containing one positive sample from $$p(x_{t+k}|c_{t})$$ and N-1 negative samples from the 'proposal' distribution $$
p(x_{t+k})
$$
The final loss(InfoNCE loss) is:$$
\mathcal{L}_{N} = - \mathbb{E}_{X} [ log \frac{f_{k}(x_{t+k},c_{t})}{\sum_{x_{j} \in X} f_{k}(x_{j},c_{t})}]
$$

A provement is given in the paper that optimizing InfoNCE loss is optimizing a **lower bound** on the mutual information between x and c.

After finishing traing by InfoNCE loss, the $$g_{enc}$$ is trained optimally so that it can be utilized in many downstream tasks.




