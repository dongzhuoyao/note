---
layout: draft
title: "Unsupervised learning"
date: 2020-03-31 14:49:0 +0000
comments: False
share: False
categories: cv
---
<!--

https://www.zhihu.com/question/355779873

https://loewex.github.io/GreedyInfoMax.html


-->

check [zhihu discussion](https://www.zhihu.com/question/355779873)



## LeCun's opinion between self-supervised learning and unsupervised learning

[https://www.facebook.com/yann.lecun/posts/10155934004262143](https://www.facebook.com/yann.lecun/posts/10155934004262143)

[Visual Task Adaptation Benchmark](https://google-research.github.io/task_adaptation/)

Similarly, representations may be pre-trained on any data, VTAB permits supervised, unsupervised, or other pre-training strategy. There is one constraint: the evaluation datasets must not be used during pre-training. This constraint is designed to mitigate overfitting to the evaluation tasks.


## [K-pair Loss: Improved Deep Metric Learning with Multi-class N-pair Loss Objective,NIPS16](https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective)


## CPC

## [CPC v2](https://arxiv.org/pdf/1905.09272.pdf)

## [Contrastive Representation Distillation,ICLR20](https://hobbitlong.github.io/CRD/)

**AMDIM**

**[MoCo: Momentum Contrast for Unsupervised Visual Representation Learning,CVPR20](https://arxiv.org/abs/1911.05722)**


![](imgs/moco.png)

Preliminary: InfoNCE paper.


A contrastive loss function, called InfoNCE, is considered in this paper. Contrastive loss functions can also be based on other forms , such as margin-based losses and variants of NCE losses.

dictionary size can be much larger than a typical mini-batch size, and can be flexibly and independently set as a hyper-parameter.

momentum update is  on **the parameter of the encoder**, not the feature of samples.


In experiments, a relatively large momentum (e.g., m = 0.999, our default) works much better than a smaller value (e.g., m = 0.9), suggesting that a slowly evolving key encoder is a core to making use of a queue.  The temperature  in InfoNCE in is set as 0.07. We set query length K = 65536.

In experiments, we found that using BN prevents the model from learning good representations, as similarly reported in [35] (which avoids using BN). The model appears to “cheat” the pretext task and easily finds a low-loss solution. This is possibly because the intra-batch communication among samples (caused by BN) leaks information. check **A.9. Ablation on Shuffling BN**

 MoCo V1's focus is on a mechanism for general contrastive learning; we do not explore orthogonal factors (such as specific pretext tasks) that may further improve accuracy. As an example, “MoCo v2”, an extension of a preliminary version of this manuscript, achieves 71.1% accuracy with R50 (up from 60.6%), given small changes on the data augmentation and output projection head. We believe that this additional result shows the generality and robustness of the MoCo framework.



MoCo v1/2 is also useful in CURL which finds MoCo "is extremely useful in Deep RL" (quote Aravind Srinivas, CURL author).<CURL: Contrastive Unsupervised Representations for Reinforcement Learning>




<div id="fb-root"></div>
<script async defer crossorigin="anonymous" src="https://connect.facebook.net/en_US/sdk.js#xfbml=1&version=v6.0"></script>

<div class="fb-post" data-href="https://www.facebook.com/hekaiming/posts/10158594924257150" data-show-text="true" data-width=""><blockquote cite="https://developers.facebook.com/hekaiming/posts/10158594924257150" class="fb-xfbml-parse-ignore"><p>Happen to see this nice video introducing MoCo v1/v2! It also covers Berkeley&#039;s recent work on CURL which finds MoCo &quot;is extremely useful in Deep RL&quot; (quote Aravind Srinivas, CURL author).</p>Posted by <a href="#" role="button">Kaiming He</a> on&nbsp;<a href="https://developers.facebook.com/hekaiming/posts/10158594924257150">Wednesday, April 15, 2020</a></blockquote></div>


## SimCLR

## Greedy InfoMax


How Useful is Self-Supervised Pretraining for Visual Tasks? see observation from [CVPR20](https://arxiv.org/abs/2003.14323)

**[Self-Supervised Learning of Video-Induced Visual Invariances,CVPR20](https://arxiv.org/pdf/1912.02783.pdf)**


The proposed unsupervised models (VIVIEx(4) / VIVI-Ex(4)-Big) trained on raw **Youtube8M** videos
and variants co-trained with 10%/100% labeled ImageNet
data (VIVI-Ex(4)-Co(10%) / VIVI-Ex(4)-Co(100%)), outperform the corresponding unsupervised (Ex-ImageNet),
semi-supervised (Semi-Ex-10%) and fully supervised (Sup100%, Sup-Rot-100%) baselines by a large margin.


shot supervision use augmentation as supervison.

video supervison use binary order prediction(forward or backward).








