---
layout: draft
title: "attention mechanism"
date: 2020-03-30 14:49:0 +0000
comments: False
share: False
categories: cv
---
<!--
todo: https://zhuanlan.zhihu.com/p/33345791
-->

## Variants

**[A2-nets-double-attention-networks,NIPS](https://papers.nips.cc/paper/7318-a2-nets-double-attention-networks.pdf)**

- global average pooling is used in the gathering process, while the
resulted single global feature is distributed to all locations, ignoring different needs across locations.
Seeing these shortcomings, we introduce this genetic formulation and propose the Double Attention
block.

check demo code: [https://github.com/gjylt/DoubleAttentionNet](https://github.com/gjylt/DoubleAttentionNet)

[colab demo](https://colab.research.google.com/drive/1sHSWP9Z_dTLH3hkJ5rbdJeWwpQ4GfK9G)

check figure tensor size illustration: [drive google](https://docs.google.com/presentation/d/1oeBbvqvvzddo6G6j43NaYSW6u0dZrxkMc1yoJS02sgk/edit?usp=sharing).

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

**[SKNet,CVPR19](https://arxiv.org/pdf/1903.06586.pdf)**

 select kernel between $$3 \times 3$$ and $$5 \times 5$$.

**[ECA-Net,CVPR20](https://arxiv.org/abs/1910.03151)**

 A improvement based on SENeti with less parameter and better performance. Check Fig 2. do k-d convolution along feature dimension$$1 \times 1 \times C$$ is a little weird for me, there should be no concept of neibourhood in feature dimension.

**[Bilinear Attention Networks,NIPS18](https://arxiv.org/pdf/1805.07932.pdf)**

**[Compact Generalized Non-local Network,NIPS](https://papers.nips.cc/paper/7886-compact-generalized-non-local-network.pdf)**

[Stand-Alone Self-Attention in Vision Models,NIPS19](https://arxiv.org/pdf/1906.05909.pdf)

[GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond,ICCV19](https://arxiv.org/abs/1904.11492)

[Exploring Self-attention for Image Recognition,arxiv](http://vladlen.info/papers/self-attention.pdf)


[BAM: Bottleneck Attention Module,BMVC18](https://arxiv.org/abs/1807.06514)

[Non-local block,CVPR18](https://arxiv.org/abs/1711.07971)

[SENet,CVPR18](https://arxiv.org/abs/1709.01507)


## Applications

Few-shot detection: [Context-Transformer: Tackling Object Confusion for Few-Shot Detection,AAAI20](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-YangZ.2509.pdf)
