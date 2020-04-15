---
layout: draft
title: "Weakly-supervised downstream tasks"
date: 2020-03-22 14:49:0 +0000
comments: False
share: False
categories: cv
---

## Weakly-supervised object detection

**[ADL,Attention-based Dropout Layer for Weakly Supervised Object Localization,CVPR19](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8954302&tag=1)**

propose an Attention-based Dropout Layer (ADL), which utilizes the self-attention mechanism to process the feature maps of the model. The proposed method is composed of two key components: 1) hiding the most
discriminative part from the model for capturing the integral extent of object, and 2) highlighting the informative region for improving the recognition power of the model. ADL is an auxiliary module which is applied only during training. During the testing phase, ADL is deactivated.


**[Adversarial Complementary Learning for Weakly Supervised Object Localization(CVPR18)](https://arxiv.org/abs/1804.06962)**

an intuitive,simple,elegant way to do adversarial erasing, but only focused on simple dataset: ILSVRC, Caltech256 and CUB-200-2011(mostly contain single object per image). How about three classifiers? can the performance be further boosted?

#### [Evaluating Weakly Supervised Object Localization Methods Right(CVPR20)](https://github.com/clovaai/wsolevaluation): this task is ill.

## Weakly-supervised image segmentation

For a quicker understanding of this topic, check this survey-style paper [A Comprehensive Analysis of Weakly-Supervised Semantic
Segmentation in Different Image Domains](https://arxiv.org/pdf/1912.11186.pdf).

#### (SEC)Seed, expand and constrain: Three principles for weakly-supervised image segmentation(ECCV16)

#### Weakly-Supervised Semantic Segmentation Network with Deep Seeded Region Growing(CVPR18)

Deep seeded region growing(DSRG) and Boundary loss. see Fig 2. similar work to SEC.


## co-segmentation, co-detection, [co-saliency detection](https://hzfu.github.io/proj_cosal_review.html)



## (multiple images) common-object localization

#### [Co-localization in Real-World Images(CVPR14)](http://vision.stanford.edu/pdf/tang14.pdf)




