---
layout: draft
title: "Attention mechanism"
permalink: /attention_mechanism
date: 2020-03-30 14:49:0 +0000
comments: False
share: False
categories: cv
tags:   [ First Tag, Second Tag,    Third Tag ]
---
<!--
todo: 
-->

[https://zhuanlan.zhihu.com/p/33345791](https://zhuanlan.zhihu.com/p/33345791)

[https://zhuanlan.zhihu.com/p/106662375](https://zhuanlan.zhihu.com/p/106662375)

[https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)



**[Non-local block,CVPR18](https://arxiv.org/abs/1711.07971)**

- Why rescale with $$\frac{1}{\sqrt{512}}$$
- Why Layer Norm

Relation between FC layer and non-local block 


Relation between self-attention and non-local block.

Consider 1D-nonlocal, FC layer can be seen as a matrix multiplication


Relation between gram matrix and non-local block.

check [https://arxiv.org/pdf/1701.01036.pdf](https://arxiv.org/pdf/1701.01036.pdf)


**[Analyzing Multi-Head Self-Attention:
Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned,ACL19](https://arxiv.org/pdf/1905.09418.pdf)**

Most of the heads can be removed by the stochastic gates.

Gumbel sofmax,TODO

**[SYNTHESIZER:Rethinking Self-Attention in Transformer Models,Arxiv2005](https://arxiv.org/pdf/2005.00743.pdf)**

Replace $$Q(x)K(x)^{T}$$ as direction function F(x) mapping from d to l.




**[See More, Know More: Unsupervised Video Object Segmentation With Co-Attention Siamese Networks,CVPR19](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lu_See_More_Know_More_Unsupervised_Video_Object_Segmentation_With_Co-Attention_CVPR_2019_paper.pdf)**

![](/imgs/co-attention.png)

[code](https://github.com/carrierlxk/COSNet/blob/master/deeplab/siamese_model_conf.py#L264)


**[Cross Attention Network for Few-shot Classification,NeuIPS19](https://papers.nips.cc/paper/8655-cross-attention-network-for-few-shot-classification.pdf)**

[Review](shttp://papers.nips.cc/paper/8655-cross-attention-network-for-few-shot-classification)

[code](https://github.com/dongzhuoyao/fewshot-CAN/blob/master/torchFewShot/models/cam.py)


For the fusion layer, the input is WH x H x W. It seems that they try to attend second attention, between WH x WH via 2D convolution. And this module can also be injected in Non-local Block. Overall, you can see the shadow of non-local block in this paper. Some differences I feel are:

- multi-pairs images 
- fusion layer to **further** reason the difference between WHxWH, except the torch.matmul(WHxC,CxWH) in Nonlocal. After all operations, a softmax is appended to finalize the whole attention mechanism.
- You can even see a residual connection after the fusion layer, which is also observsed in non-local block!
- For non-local block, it's not bidirectional, while the author's network is bidirectional. They can influence each other because the correlation layer generates two mutual-transposed matrix via torch.matmul(WHxC,CxWH).

![](/imgs/cross-attention1.png)
![](/imgs/cross-attention2.png)

**[Effective Approaches to Attention-based Neural Machine Translation,EMNLP15](https://arxiv.org/pdf/1508.04025.pdf)**


Global vs. Local Attention

**[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention,ICML15]()**

introduce the hard and soft attention.

**[SENet,CVPR18](https://arxiv.org/abs/1709.01507)**

## Variants

**[Multi Head Attention,NeuIPS17](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)**

Concatenated the result of $$softmax(\frac{QK^{T}}{\sqrt{n}} V)$$ and send them into a linear layer to remap back to original shape.s

**[A2-nets-double-attention-networks,NIPS18](https://papers.nips.cc/paper/7318-a2-nets-double-attention-networks.pdf)**

For SENet, global average pooling is used in the gathering process, while the
resulted single global feature is distributed to all locations, ignoring different needs across locations.
Seeing these shortcomings, we introduce this genetic formulation and propose the Double Attention
block.

check demo code: [https://github.com/gjylt/DoubleAttentionNet](https://github.com/gjylt/DoubleAttentionNet)

[colab demo](https://colab.research.google.com/drive/1sHSWP9Z_dTLH3hkJ5rbdJeWwpQ4GfK9G)

[NeuIPS review](https://media.nips.cc/nipsbooks/nipspapers/paper_files/nips31/reviews/233.html)


check figure tensor size illustration: [drive google](https://docs.google.com/presentation/d/1oeBbvqvvzddo6G6j43NaYSW6u0dZrxkMc1yoJS02sgk/edit?usp=sharing).

![](/imgs/DoubleAttentionNetwork_NIPS.png)

**[Graph-Based Global Reasoning Networks,CVPR19](https://arxiv.org/pdf/1811.12814.pdf)**


check [tensor size fig](https://docs.google.com/presentation/d/1AlmSFH0C00f74pgYOV5VaCAdTDuZcg6AwoLVN3huOhA/edit?usp=sharing)

Different from the recently proposed Non-local Neural Networks (NL-Nets)  and Double Attention Networks  which only focus on delivering information and
rely on convolution layers for reasoning, our proposed
model is able to directly reason on relations over regions.
Similarly, Squeeze-and-Extension Networks (SE-Nets) 
only focus on incorporating image-level features via global
average pooling, leading to an interaction graph containing
only one node. It is not designed for regional reasoning
as our proposed method. Extensive experiments show that
inserting our GloRe can consistently boost performance of
state-of-the-art CNN architectures on diverse tasks including image classification, semantic segmentation and video action recognition.

![](../imgs/global_reasoning_unit.png)


**[CBAM，ECCV18](https://arxiv.org/pdf/1807.06521.pdf)**

use max pooling and avgpooling together. 
<!--sss work-->

**[An Empirical Study of Spatial Attention Mechanisms in Deep Networks,ICCV19](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhu_An_Empirical_Study_of_Spatial_Attention_Mechanisms_in_Deep_Networks_ICCV_2019_paper.pdf)**

**[CCNet: Criss-Cross Attention for Semantic Segmentation,ICCV19](https://arxiv.org/pdf/1811.11721.pdf)**

- Two crisscross attention modules before and after share the same parameters to avoid adding too many extra parameters. 
- attention matrix is $$(H+W−1)\times WH$$, softmax is applied along $$W+H-1$$ axis.

**[Local Relation Networks for Image Recognition,ICCV19](https://arxiv.org/pdf/1904.11491.pdf)**

Check Fig 2.

> use non local networks (i.e self attention) to help compute local relations? 




**[Dynamic Graph Message Passing Networks](https://arxiv.org/pdf/1909.05235.pdf)**

quite similart o deformable convolution. A fundamental difference to deformable convolution is
that it only learns the offset dependent on the input feature
while the filter weights are fixed for all inputs. In contrast,
our model learns the random walk, weight and affinity as
all being dependent on the input. This property makes our
weights and affinities position-specific whereas deformable
convolution shares the same weight across all convolution
positions in the feature map. learns to sample
a set of K nodes (where K  9) for message passing globally from the whole feature map. This allows our model to
capture a larger receptive field than deformable convolution.


**[Adaptive Pyramid Context Network for Semantic Segmentation,CVPR19](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Adaptive_Pyramid_Context_Network_for_Semantic_Segmentation_CVPR_2019_paper.pdf)**

unlike non-local with three inputs, the moudule here has only two input. Check Fig 2. Reshaping from $$H\times W \times S^{2}$$ to $$HW \times S^{2}$$ for forming affinity matrix is a little counter-intuitive. 

divide the feature map X of image I into s×s subregions, running adaptive pooling to generate 1x1,2x2,3x3,4x4... region pooled features.

No softmax is applied around affinity matrix.



![](/imgs/adaptive-context-module.png)

**[Long-Term Feature Banks for Detailed Video Understanding,CVPR19](https://arxiv.org/abs/1812.05038)**

![](/imgs/long-term-feature-bank.png)

**[Asymmetric Non-local Neural Networks for Semantic Segmentation,ICCV19](https://arxiv.org/pdf/1908.07678.pdf)**

check Fig 1.


**[CARAFE: Content-Aware ReAssembly of FEatures,ICCV19](https://arxiv.org/pdf/1905.02188.pdf)**


CARAFE can be seamlessly integrated into existing
frameworks where upsampling operators are needed

CARAFE works as a reassembly operator with contentaware kernels. It consists of two steps. The first step is to
predict a reassembly kernel for each target location according to its content, and the second step is to reassemble the features with predicted kernels.

check [code](https://github.com/XiaLiPKU/CARAFE/blob/master/carafe.py)

**[Spatial Pyramid Based Graph Reasoning for Semantic Segmentation,CVPR20](https://arxiv.org/pdf/2003.10211.pdf)**



**[SKNet,CVPR19](https://arxiv.org/pdf/1903.06586.pdf)**

 select kernel between $$3 \times 3$$ and $$5 \times 5$$.

 [code repo](https://github.com/pppLang/SKNet/blob/master/sknet.py)

 ![](/imgs/sknet.png)

 [source file](https://docs.google.com/presentation/d/18ewlZqw8RR0t_ElRRSrEI9raykxgsuThKnv27DWhlUE/edit?usp=sharing)

 **[ResNeSt: Split-Attention Networks,Arxiv2002](https://hangzhang.org/files/resnest.pdf)**

As in ResNeXt blocks, the input feature-map can
be divided into several groups along the **channel** dimension, and the number of
feature-map groups is given by a **cardinality** hyperparameter K. We refer to the
resulting feature-map groups as cardinal groups. We introduce a new **radix** hyperparameter R that dictates the number of splits within a cardinal group. Then the block input X are split into G = KR groups along the channel dimension.

Can be seen as a combination of ResNext and SKNet.

![](/imgs/resnest.png)


**[Improving Convolutional Networks with Self-Calibrated Convolutions,CVPR20](https://github.com/backseason/SCNet)**



**[ECA-Net,CVPR20](https://arxiv.org/abs/1910.03151)**

 An improvement based on SENet with less parameter and better performance. Check Fig 2. do 1D convolution with kernel size k along feature dimension$$1 \times 1 \times C$$ is a little counter-intuitive for me, there should be no concept of 'neibourhood' in feature dimension.

**[DANet:Dual Attention Network for Scene Segmentation(CVPR2019)](https://arxiv.org/abs/1809.02983)**

**[Bilinear Attention Networks,NIPS18](https://arxiv.org/pdf/1805.07932.pdf)**

**[Compact Generalized Non-local Network,NIPS](https://papers.nips.cc/paper/7886-compact-generalized-non-local-network.pdf)**

**[Stand-Alone Self-Attention in Vision Models,NIPS19](https://arxiv.org/pdf/1906.05909.pdf)**

**[GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond,ICCV19](https://arxiv.org/abs/1904.11492)**

**[Exploring Self-attention for Image Recognition,arxiv](http://vladlen.info/papers/self-attention.pdf)**


**[BAM: Bottleneck Attention Module,BMVC18](https://arxiv.org/abs/1807.06514)**

![](/imgs/BAM.png)



## Applications

Few-shot detection: [Context-Transformer: Tackling Object Confusion for Few-Shot Detection,AAAI20](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-YangZ.2509.pdf)

**[Self-Attention Generative Adversarial Networks](https://arxiv.org/pdf/1805.08318.pdf)**

copy non-local block into GAN.

**[Actor-Transformers for Group Activity Recognition,CVPR20](https://isis-data.science.uva.nl/cgmsnoek/pub/gavrilyuk-transformers-cvpr2020.pdf)**

Group Activity Recognition: use non-local block to fuse optical flow, pose, RGB. check Fig 2.

## Gate Mechanism

**RNN**


[https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)



**LSTM**

[https://en.wikipedia.org/wiki/Long_short-term_memory](https://en.wikipedia.org/wiki/Long_short-term_memory)

[https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation. The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means “let nothing through,” while a value of one means “let everything through!” An LSTM has three of these gates, to protect and control the cell state.

intuition: tanh (to push the values to be between −1 and 1) and multiply it by the output of the sigmoid( to push the values to be between 0 and 1), so that we only output the parts we decided to.s