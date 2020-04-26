---
layout: draft
title: "Unsupervised learning"
permalink: /unsupervised_learning
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



## Mutual Information

I(X;Y)=0 if X and Y is unrelated. otherwise I(X;Y) is larger than zero.

Intuitively, mutual information measures the information that X and Y share: It measures how much knowing one of these variables reduces uncertainty about the other. If X=Y+3 or X=Y^3, which means X is a deterministic function of Y and vice versa, then mutual information I(X;Y) is the entropy of Y(or X).

**Property**:  Non-negativity, and Symmetry.

![Venne gram](imgs/information-measure.png)

TO CONTINUE




**LeCun's opinion between self-supervised learning and unsupervised learning**

[https://www.facebook.com/yann.lecun/posts/10155934004262143](https://www.facebook.com/yann.lecun/posts/10155934004262143)

[Visual Task Adaptation Benchmark](https://google-research.github.io/task_adaptation/)

Similarly, representations may be pre-trained on any data, VTAB permits supervised, unsupervised, or other pre-training strategy. There is one constraint: the evaluation datasets must not be used during pre-training. This constraint is designed to mitigate overfitting to the evaluation tasks.



**[Noise Contrastive Estimation](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)**

A broadly used method in NLP, also used in [CPC: Representation Learning with Contrastive Predictive Coding,Arxiv18](https://arxiv.org/abs/1807.03748), [Unsupervised Feature Learning via Non-Parametric Instance Discrimination,CVPR18](https://arxiv.org/pdf/1805.01978.pdf), etc.

Originally proposed in 2010, used in NLP from [this paper: A fast and simple algorithm for training neural probabilistic language models](https://www.cs.toronto.edu/~amnih/papers/ncelm.pdf). In NLP, NCE fixes the value of normalizer Z, making the inference (if you need probability values) faster. You don't need to sum over all the vocabulary to get the normalizer value.


The reason why NCE loss will work is because NCE approximates maximum likelihood estimation (MLE) when the ratio of noise to real data 𝑘 increases.


Negative Sample is a special case of NCE, for reason you can check the [summary of approximating softmax](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650720050&idx=2&sn=9fedc937d3128462c478ef7911e77687&chksm=871b034cb06c8a5a8db8a10f708c81025fc62084d871ac5d184bab5098cb64e939c1c23a7369&mpshare=1&scene=1&srcid=0613xBLYGgZUw99YG99QMP6p#rd), also you can check a detailed comparison between negative sample and NCE in [this note: Notes on Noise Contrastive Estimation and Negative Sampling](https://arxiv.org/pdf/1410.8251.pdf)



**[A fast and simple algorithm for training neural probabilistic language models](https://www.cs.toronto.edu/~amnih/papers/ncelm.pdf)**

Motivation of this work is based on the following points:

- gradient-based back propogation is a must.
- We cannot sum over all vocabulary to get the normalizer value, then we need to approximate it.
- Why not approximate it from the perspective of gradient?
- softmax classifier summarize the classification problem as a N-class problem, NCE summarizes the classification problem as a binary-class problem. Even though, the gradient derivation pattern is similar.

Equation 10 can be derived by normal derivation rule.

Equation 11 is from the definition of expectation. 

TODO, why Equation 12 is the Maximum Likelihood gradient



**[Unsupervised Feature Learning via Non-Parametric Instance Discrimination,CVPR18](https://arxiv.org/pdf/1805.01978.pdf)**

[pytorch code](https://github.com/zhirongw/lemniscate.pytorch)

maintain a image-levell memory bank. Check Fig 2.

use noise-contrastive estimation[NCE] 

The conceptual change from class weight vector wj to feature representation vj directly is significant.
The weight vectors {wj} in the original softmax formulation are only valid for training classes. Consequently, they are not generalized to new classes, or in our setting, new instances. When we get rid of these weight vectors, our learning objective focuses entirely on the feature representation and its induced metric, which can be applied everywhere in the space and to any new instances at the test.


Computing the non-parametric softmax above is cost prohibitive when the number of classes n(n images in the dataset) is very large, e.g. at the scale of millions.  The solution is to cast the multi(n)-class classification problem into a set of binary classification problems, where the binary classification task is to discriminate between data samples and noise samples. 


**[K-pair Loss: Improved Deep Metric Learning with Multi-class N-pair Loss Objective,NIPS16](https://papers.nips.cc/paper/6200-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective)**


**[Contrastive Representation Distillation,ICLR20](https://hobbitlong.github.io/CRD/)**



**[CMC: Contrastive Multiview Coding](http://people.csail.mit.edu/yonglong/yonglong/cmc_icml_workshop.pdf)**

an extension based on CPC.

check session 2.2



**[CPC: Representation Learning with Contrastive Predictive Coding,Arxiv18](https://arxiv.org/abs/1807.03748)**


**The key insight** of CPC is to learn such representations by predicting the future in latent
space by using powerful autoregressive models.


Check [NIPS invited talks here](https://slideslive.com/38922758/invited-talk-contrastive-predictive-coding)

For a brief summary, you can check the session 2 of [this paper](https://arxiv.org/pdf/1905.11786.pdf).


![](/imgs/cpc.png)

### Basic idea

latent representations $$z_{t}=g_{enc}(x_{t})$$. An autoregressive model $$g_{ar}$$ summarizes all z<=t  in the latent space and produces a context latent representation $$c_{t}=g_{ar}(z\le t)$$.

Mutual information between original signal x and c: 
$$ 
I(x;c) = \sum_{x,c} p(x,c) log \frac{p(x|c)}{p(x)} 
$$, here is another writing of mutual information because of the bayesian rule(common rule, no extra condition is needed).


predict future observations as $$\frac{p(x_{t+k}|c_{t})}{p(x_{t+k})}$$ 
rather than 
$$p_{k}(x_{t+k}|c_{t})$$

In reality, a simple log-bilinear model $$
f_{k}(x_{t+k},c_{t})=exp(z^{T}_{t+k}W_{k}c_{t})
$$ is used. The authors precise that the linear transformation $$Wc_{t}$$ can be replaced by non-linear models like neural nets.

Given X=$$x_{1},...,x_{N}$$ of N random samples containing one positive sample from $$p(x_{t+k}|c_{t})$$ and N-1 negative samples from the 'proposal' distribution $$
p(x_{t+k})
$$
The final loss(**InfoNCE loss**) is:$$
\mathcal{L}_{N} = - \mathbb{E}_{X} [ log \frac{f_{k}(x_{t+k},c_{t})}{\sum_{x_{j} \in X} f_{k}(x_{j},c_{t})}]
$$

Considering $$f_{k}(.,.)$$ is a **log**-bilinear model, InfoNCE loss can also be viewed as a softmax loss.


#### Optimizing InfoNCE loss is optimizing a **lower bound** on the mutual information between x and c.

The observation(game rule) is 1 positive sample from $$p(x_{t+k}/c_{t})$$ and N-1 negative samples from $$p(x_{t+k})$$. Each event contains 1 positive sample and N-1 negative samples, that's the rule defining an event.

Therefore, The probability of that the i-th sample is positive(meantime the other N-1 is negative) is:

$$
p(d=i|X,c_{t}) = \frac{p(x_{i}|c_{t})\prod_{l \neq i} p(x_{l})}{\sum_{j=1}^{N} p(x_{j}|c_{t}) \prod_{l \neq j} p(x_{l})}
= \frac{p(x_{i}|c_{t})/p(x_{i})}{\sum_{j=1}^{N} p(x_{j}|c_{t})/p(x_{j})}
$$


A futher provement relating InfoNCE and $$
p(d=i|X,c_{t})$$ can be seen in supplimentary file.

TODO, cannot figure out the logic from Equation 10 to Equation 11. how to induce p(x,c) is equivalant to p(x)?

After finishing traing by InfoNCE loss, the $$g_{enc}$$ is trained optimally so that it can be utilized in many downstream tasks.

**[CPCv2: Data-Efficient Image Recognition with Contrastive Predictive Coding](https://openreview.net/forum?id=rJerHlrYwH)**

[check review](https://openreview.net/forum?id=rJerHlrYwH)

**[DIM: Learning deep representations by mutual information estimation
and maximization,ICLR19,oral](https://arxiv.org/pdf/1808.06670.pdf)**

**[AMDIM:Learning Representations by Maximizing Mutual
Information Across Views]()**

self-supervised representation learning based on maximizing mutual
information between features extracted from multiple views(**multiple augmentations**) of a shared context(**same image**)

based on previous work Deep InfoMax(DIM,2019).Our model, which we call Augmented Multiscale DIM (AMDIM), extends the local version of Deep InfoMax introduced by Hjelm et al. [2019] in several ways. First, we maximize mutual information between features extracted from independently-augmented copies of each image, rather than between features extracted from a single, unaugmented copy of each image.2 Second, we maximize mutual
information between multiple feature scales simultaneously, rather than between a single global and
local scale. Third, we use a more powerful encoder architecture. Finally, we introduce mixture-based
representations. We now describe local DIM and the components added by our new model.

TODO

**[MoCo: Momentum Contrast for Unsupervised Visual Representation Learning,CVPR20](https://arxiv.org/abs/1911.05722)**


![](/imgs/moco.png)

Preliminary: InfoNCE paper.


A contrastive loss function, called InfoNCE, is considered in this paper. Contrastive loss functions can also be based on other forms , such as margin-based losses and variants of NCE losses.


Relationship with previous works: Compared to another similar idea of memory bank (Wu et al, 2018) which stores representations of all the data points in the database and samples a random set of keys as negative examples, a queue-based dictionary in MoCo enables us to reuse representations of immediate preceding mini-batches of data.

Motivation of queue setting:  removing the oldest mini-batch can be beneficial, because its encoded keys are the most outdated and thus the least consistent with the newest ones.

Motivation of moment update: Using a queue can make the dictionary large, but it also makes it intractable to update the key encoder by back-propagation (the gradient should propagate to all samples in the queue).

dictionary size can be much larger than a typical mini-batch size, and can be flexibly and independently set as a hyper-parameter.

momentum update is  on **the parameter of the encoder**, not the feature of samples.  same pattern also occurs in [mean teacher paper](https://arxiv.org/pdf/1703.01780.pdf).




In experiments, a relatively large momentum (e.g., m = 0.999, our default) works much better than a smaller value (e.g., m = 0.9), suggesting that a slowly evolving key encoder is a core to making use of a queue.  The temperature  in InfoNCE in is set as 0.07. We set query length K = 65536.

In experiments, we found that using BN prevents the model from learning good representations, as similarly reported in [35] (which avoids using BN). The model appears to “cheat” the pretext task and easily finds a low-loss solution. This is possibly because the intra-batch communication among samples (caused by BN) leaks information. check **A.9. Ablation on Shuffling BN**

 MoCo V1's focus is on a mechanism for general contrastive learning; we do not explore orthogonal factors (such as specific pretext tasks) that may further improve accuracy. As an example, “MoCo v2”, an extension of a preliminary version of this manuscript, achieves 71.1% accuracy with R50 (up from 60.6%), given small changes on the data augmentation and output projection head. We believe that this additional result shows the generality and robustness of the MoCo framework.



MoCo v1/2 is also useful in CURL which finds MoCo "is extremely useful in Deep RL" (quote Aravind Srinivas, CURL author).<CURL: Contrastive Unsupervised Representations for Reinforcement Learning>




<div id="fb-root"></div>
<script async defer crossorigin="anonymous" src="https://connect.facebook.net/en_US/sdk.js#xfbml=1&version=v6.0"></script>

<div class="fb-post" data-href="https://www.facebook.com/hekaiming/posts/10158594924257150" data-show-text="true" data-width=""><blockquote cite="https://developers.facebook.com/hekaiming/posts/10158594924257150" class="fb-xfbml-parse-ignore"><p>Happen to see this nice video introducing MoCo v1/v2! It also covers Berkeley&#039;s recent work on CURL which finds MoCo &quot;is extremely useful in Deep RL&quot; (quote Aravind Srinivas, CURL author).</p>Posted by <a href="#" role="button">Kaiming He</a> on&nbsp;<a href="https://developers.facebook.com/hekaiming/posts/10158594924257150">Wednesday, April 15, 2020</a></blockquote></div>


**[SimCLR:A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)**


check [zhihu](https://www.zhihu.com/question/372064916)

With 128 TPU v3 cores, it takes ∼1.5 hours to train our ResNet-50 with a batch size of 4096 for 100 epochs.


check suppimentary, find a detailed comparison with previous methods.

![](imgs/simclr.png)


**[On Mutual Information Maximization for Representation Learning,ICLR20](https://openreview.net/forum?id=rkxoh24FPH)**

check concolusions.

 it is unclear whether the connection to MI is a sufficient (or
necessary) component for designing powerful unsupervised representation learning algorithms. We
propose that the success of these recent methods could be explained through the view of triplet-based
metric learning and that leveraging advances in that domain might lead to further improvements. 


**[Greedy InfoMax:Putting An End to End-to-End:Gradient-Isolated Learning of Representations,NIPS19](https://arxiv.org/pdf/1905.11786.pdf)**

[NIPS proceeding link](http://papers.nips.cc/paper/8568-putting-an-end-to-end-to-end-gradient-isolated-learning-of-representations)

[code](https://github.com/loeweX/Greedy_InfoMax)



Gradient is isolated between modules, but still need gradient in modules.

Each module will generate a CPC loss.



![](imgs/greedy-infomax.png)


How Useful is Self-Supervised Pretraining for Visual Tasks? see observation from [CVPR20](https://arxiv.org/abs/2003.14323)

**[Self-Supervised Learning of Video-Induced Visual Invariances,CVPR20](https://arxiv.org/pdf/1912.02783.pdf)**


The proposed unsupervised models (VIVIEx(4) / VIVI-Ex(4)-Big) trained on raw **Youtube8M** videos
and variants co-trained with 10%/100% labeled ImageNet
data (VIVI-Ex(4)-Co(10%) / VIVI-Ex(4)-Co(100%)), outperform the corresponding unsupervised (Ex-ImageNet),
semi-supervised (Semi-Ex-10%) and fully supervised (Sup100%, Sup-Rot-100%) baselines by a large margin.


shot supervision use augmentation as supervison.

video supervison use binary order prediction(forward or backward).


## Self-supervised learning for video correspondence

**[Learning Correspondence from the Cycle-consistency of Time,](https://arxiv.org/pdf/1903.07593.pdf)**


**[Self-supervised Learning for Video Correspondence Flow,BMVC19](https://arxiv.org/pdf/1905.00875.pdf)**


**[MAST: A Memory-Augmented Self-Supervised Tracker,CVPR20](https://arxiv.org/pdf/2002.07793.pdf)**

<!--
https://zhuanlan.zhihu.com/p/125506819
-->







