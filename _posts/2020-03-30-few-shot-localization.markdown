---
layout: draft
title: "Few-shot  localization"
date: 2020-03-30 14:49:0 +0000
comments: False
share: False
categories: cv
---

## Few-shot segmentation

**[One-Shot Segmentation in Clutter,ICML18](https://arxiv.org/pdf/1803.09597.pdf)**

**[CANet: Class-Agnostic Segmentation Networks with Iterative Refinement and
Attentive Few-Shot Learning,CVPR19](https://arxiv.org/pdf/1903.02351.pdf)**

Check Fig 2.

[PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment,ICCV19](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_PANet_Few-Shot_Image_Semantic_Segmentation_With_Prototype_Alignment_ICCV_2019_paper.pdf):

Masked Average Pooling + Cosine similarity, obtain final segmentation result.

**[Objectness-Aware One-Shot Semantic Segmentations,Arxiv20,April](https://arxiv.org/pdf/2004.02945.pdf)**

- adopt HRNetV2-W48 as the backbone of the objectness module. 
- The objectness module is trained to segment out all objects in the image.(train the objectness module for 300,000 iterations with batch size 4, which takes about 50 hours on GeForce GTX 1080Ti. )
- Check Fig 2, support feature, query feature, and objectness feature are congregated by adding operation.


## Few-shot detection

**[Weakly Supervised Few-shot Object Segmentation using Co-Attention with Visual and Semantic Inputs,Arxiv20](https://arxiv.org/pdf/2001.09540.pdf)**

only requiring image-level classification data for few-shot object segmentation. propose a novel multi-modal interaction module for few-shot object segmentation that utilizes a coattention mechanism using both visual and word embedding.

Class wording embedding is then spatially tiled and concatenated with the visual features resulting in flattened matrix representations. 

Unlike non-local block relating $$WH \times C$$ and $$C \times WH$$, they add an extra $$C \times C$$ matrix in the very middle. Also, they consider two-directions by applying softmax along different dimensions. Check Fig 2.




[Few-Shot Object Detection with Attention-RPN and Multi-Relation Detector,CVPR20](https://arxiv.org/abs/1908.01998https://arxiv.org/abs/1908.01998)

[Context-Transformer: Tackling Object Confusion for Few-Shot Detection,AAAI20](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-YangZ.2509.pdf)

## Few-shot instance segmentation

[One-Shot Instance Segmentation,Arxiv](https://arxiv.org/pdf/1811.11507.pdf)

[Differentiable Meta-learning Model for Few-shot Semantic Segmentation,AAAI20](https://arxiv.org/pdf/1911.10371.pdf)

[FGN: Fully Guided Network for Few-Shot Instance Segmentation,CVPR20](https://arxiv.org/abs/2003.13954)

## Few-shot Edge Detection

[CAFENet: Class-Agnostic Few-Shot
Edge Detection Network,Arxiv](https://arxiv.org/pdf/2003.08235.pdf)

## Few-shot video activity localization 

**[METAL: Minimum Effort Temporal Activity Localization in Untrimmed Videos,CVPR20](https://sites.cs.ucsb.edu/~yfwang/papers/cvpr2020.pdf)**






