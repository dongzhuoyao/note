---
layout: draft
title: "Classification, segmentation, detection"
date: 2020-03-22 14:49:0 +0000
comments: False
share: False
categories: ml
---

## Detection

#### Anchor-free

- [CenterNet](https://arxiv.org/pdf/1904.07850.pdf): no anchor any more; we only have one positive “anchor” per object, and hence do not need NonMaximum Suppression (NMS);a larger output resolution (output stride of 4) compared to traditional object detectors(output stride of 16). We use a single network to predict
the keypoints Yˆ , offset Oˆ(recover the discretization error caused by the output
stride,), and size Sˆ(regress the width and width of bboxes). The network predicts a total of C + 4 outputs at each location. All outputs share a common fully-convolutional backbone network. Compared with **CornerNet,ExtremeNet**, they require a combinatorial grouping stage after keypoint detection, which significantly slows
down each algorithm.
- [CornerNet,ECCV18](https://arxiv.org/pdf/1808.01244.pdf):A convolutional network outputs a heatmap
for all top-left corners, a heatmap for all bottom-right corners, and an embedding vector for each detected corner. The network is trained to predict similar embeddings for corners that belong to the same object.
- [ExtremeNet,CVPR19](https://arxiv.org/pdf/1901.08043.pdf):
  

#### Different det heads

check [related work part](https://arxiv.org/pdf/1904.06493.pdf), 

- Faster-RCNN, $$1024\times 7\times 7$$
- [Light-head RCNN,Arxiv](https://arxiv.org/pdf/1711.07264.pdf): generate the feature maps with small channel number (thin feature maps) 490 (10 × 7 × 7), kernel size=15, Cmid=64, Cout=490(10x7x7),**followed by** conventional RoI warping; large kernel+seperable convolution.
- [R-FCN](https://arxiv.org/pdf/1605.06409.pdf): 3969 (81 × 7 × 7), $$k^{2}(C+1)\times W\times H$$  after RoI pooling obtain $$k^{2}(C+1) \times 7 \times 7$$, check Fig 2. Aside from the above $$k^{2}(C +1)$$ convolutional layer for bbox classification, we append a sibling $$4k^{2}$$ convolutional layer for bounding box regression. The position-sensitive RoI pooling is performed on this bank of $$4k^{2}$$ maps, producing a $$4k^{2}$$ vector for each RoI. Then it is aggregated into a 4-d vector by average voting. Noticeably, there is **no learnable layer** after the RoI layer, enabling nearly cost-free region-wise computation and speeding up both training and inference. Similar idea in segmentation is FCIS,instanceFCN.
- [Double head,CVPR20](https://arxiv.org/pdf/1904.06493.pdf): check Fig 1.
- [Cascaded-RCNN,CVPR18](https://arxiv.org/pdf/1712.00726.pdf): It consists of a sequence of detectors trained with increasing IoU thresholds, to be sequentially more selective against close false positives. see Fig 3.
- [IoUNet,ECCV18](https://arxiv.org/abs/1807.11590). Fig 2 proves classfication score is not enough for det, and localization score really exists. IoU estimator can be used as an early-stop condition to implement iterative refinement with adaptive steps.
- [Mask Scoring RCNN,CVPR19](https://arxiv.org/pdf/1903.00241.pdf): similar intuition as IoUNet. in most instance segmentation pipelines, such as Mask R-CNN  and MaskLab, the score of the instance mask is shared with box-level classification confidence, which is predicted by a classifier applied on the proposal feature. It is inappropriate to use classification confidence to measure the mask quality since it only serves for distinguishing the semantic categories of proposals, and is not aware of the actual quality and completeness of the instance mask. The paper focuses on designing an extra head to predict mask score.

### RoI Pooling

- RoI Pooling
- RoI Align
- PrRoI Pooling

#### NMS

- IoU-NMS from IoUNet
- Soft-NMS
- learning to NMS
  

## Instance Segmentation
<!--https://www.zhihu.com/question/360594484-->

#### [SOLOv2](https://arxiv.org/pdf/2003.10152.pdf):

## Classification

## Segmentation

#### [PSANet,ECCV18](https://hszhao.github.io/papers/eccv18_psanet.pdf)



#### [Context Prior,CVPR20](https://arxiv.org/pdf/2004.01547.pdf)
Learn a $$WH \times WH$$ affinity matrix,
k=11 in context aggregation is vital for the functionality of Context Prior, without k=11, CP cannot work.



- 


## Reference

- [colab](https://colab.research.google.com/drive/1US3uQNTWUse1-D_4oK5TlKBRfAbrZmxD)
- [https://zhuanlan.zhihu.com/p/83517817](https://zhuanlan.zhihu.com/p/83517817)
- [https://zhuanlan.zhihu.com/p/56924766](https://zhuanlan.zhihu.com/p/56924766)

