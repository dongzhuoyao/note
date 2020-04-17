---
layout: draft
title: "Classification, segmentation, detection"
permalink: /cls_seg_det
date: 2020-03-22 14:49:0 +0000
comments: False
share: False
categories: cv
---

## Detection

For main progress check [Survey 2019](https://link.springer.com/content/pdf/10.1007/s11263-019-01247-4.pdf)

#### Mainstream progress

**[EfficientDet: Scalable and Efficient Object Detection,CVPR20](https://arxiv.org/pdf/1911.09070.pdf)**

**[Dynamic R-CNN: Towards High Quality Object Detection via Dynamic Training,Arxiv2004](https://arxiv.org/pdf/2004.06002.pdf)**

Dynamic R-CNN to adjust the label assignment criteria (IoU threshold) and the shape of regression loss function (parameters of SmoothL1 Loss) automatically based on the statistics of proposals during training


#### Anchor-free

  **[CenterNet](https://arxiv.org/pdf/1904.07850.pdf)**
  
  
  no anchor any more; we only have one positive “anchor” per object, and hence do not need NonMaximum Suppression (NMS);a larger output resolution (output stride of 4) compared to traditional object detectors(output stride of 16). We use a single network to predict
the keypoints  , offset (recover the discretization error caused by the output
stride,), and size (regress the width and width of bboxes). The network predicts a total of C + 4 outputs at each location. All outputs share a common fully-convolutional backbone network. 

Compared with **CornerNet,ExtremeNet**, they require a combinatorial grouping stage after keypoint detection, which significantly slows
down each algorithm.

  **[CornerNet,ECCV18](https://arxiv.org/pdf/1808.01244.pdf)**
  
  A convolutional network outputs a heatmap
for all top-left corners, a heatmap for all bottom-right corners, and an embedding vector for each detected corner. The network is trained to predict similar embeddings for corners that belong to the same object.


**[ExtremeNet,CVPR19](https://arxiv.org/pdf/1901.08043.pdf)**
  

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


**[Spatially Attentive Output Layer for Image Classification,CVPR20](https://arxiv.org/pdf/2004.07570.pdf)**

waiting for code.


[Adversarial Examples Improve Image Recognition,CVPR20](https://arxiv.org/pdf/1911.09665.pdf)

 - propose to use two batch norm statistics, one for clean images and one auxiliary for adversarial examples. The two batchnorms properly disentangle the two distributions at normalization layers for accurate statistics estimation. We show this distribution disentangling is crucial, enabling us to successfully improve, rather than degrade, model performance with adversarial examples
- the first to show adversarial examples can improve model performance in the fully-supervised setting on the large-scale ImageNet dataset.
- a simple  auxiliary BN design, check Fig 3.

#### architecture

**[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,ICML19](https://arxiv.org/pdf/1905.11946.pdf)**

**[HRNet,PAMI20](https://arxiv.org/pdf/1908.07919.pdf)**

check Fig 2.

**[Res2Net,PAMI20](https://arxiv.org/pdf/1904.01169.pdf)**

The Res2Net strategy exposes a new dimension, namely scale
(the number of feature groups in the Res2Net block), as an
essential factor in addition to existing dimensions of depth,
width, and cardinality. 

**[DHM,CVPR20](https://arxiv.org/pdf/2003.10739.pdf)**




#### Visualization-related

CAM

[grad-CAM](https://arxiv.org/abs/1610.02391)

## Semantic Segmentation

Check the survey [here](https://arxiv.org/pdf/2001.05566.pdf).


**[PSANet,ECCV18](https://hszhao.github.io/papers/eccv18_psanet.pdf)**

**[DANet](https://arxiv.org/pdf/1809.02983.pdf)**

non-local on channel and spatial.


**[Context Prior,CVPR20](https://arxiv.org/pdf/2004.01547.pdf)**
Learn a $$WH \times WH$$ affinity matrix,
k=11 in context aggregation is vital for the functionality of Context Prior, without k=11, CP cannot work.

Affinity matrix construction is similar to **[Adaptive Pyramid Context Network for Semantic Segmentation,CVPR19](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Adaptive_Pyramid_Context_Network_for_Semantic_Segmentation_CVPR_2019_paper.pdf)**










## Instance Segmentation

**[Deep Snake for Real-Time Instance Segmentation,CVPR20,oral](https://arxiv.org/pdf/2001.01629.pdf)**












## Human-object interaction

[Spatial Priming for Detecting Human-Object Interactions,Arxiv2004](https://arxiv.org/pdf/2004.04851.pdf)










## Batch Normalization

check related work in [ICLR20](https://arxiv.org/pdf/2001.06838.pdf).









## Upsampling 

[Understanding Convolution for Semantic Segmentation,WACV18](https://arxiv.org/abs/1702.08502)










## Pooling

#### Bilinear Pooling


**[Bilinear CNN](http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf)**

check code for understading: [https://github.com/HaoMood/bilinear-cnn/blob/master/src/bilinear_cnn_all.py#L71](https://github.com/HaoMood/bilinear-cnn/blob/master/src/bilinear_cnn_all.py#L71)


**[](http://openaccess.thecvf.com/content_ICCV_2017/papers/Li_Factorized_Bilinear_Models_ICCV_2017_paper.pdf)**
Although the bilinear pooling is capable of capturing pairwise interactions, it also introduces a quadratic
number of parameters in weight matrices $$W^{R}_{j}$$ leading to
huge computational cost and the risk of overfitting.

$$
y = b + w^{T}x +x^{T}F^{T}Fx
$$

where $$R \in \mathbb{R}^{k \times n}$$

The key idea of our DropFactor is to randomly drop the bilinear paths corresponding to k factors during the training.

performs nearly the same with bilinear pooling in fine-grained classification task. therefore the author mainly focus on the experimental of traditional classifcation task. check last subsession.



**[MPN-COV](https://arxiv.org/pdf/1703.08050.pdf)**




#### [Strip Pooling,CVPR20](https://arxiv.org/pdf/2003.13328.pdf)

Check Fig2, the spatial dimension doesn't change, why name it ``pooling''?

Unlike the two-dimensional average pooling, the proposed strip pooling averages all the feature values in a row or a column. Given the horizontal and vertical strip pooling layers, it is easy to build long-range dependencies between regions distributed discretely and encode regions with the banded shape, thanks to the long and narrow kernel shape.

check implementation [here](https://github.com/Andrew-Qibin/SPNet/blob/master/models/customize.py).



#### Spatial Pyramid Pooling

By adopting a set of parallel pooling operations with a unique kernel size at each pyramid level, the network is able to capture largerange context. It has been shown promising on several scene parsing benchmarks.its ability to exploit contextual information is limited since only square kernel shapes are applied. Moreover, the spatial pyramid pooling is only modularized on top of the backbone network thus rendering it is not flexible or directly applicable in the network building block for feature learning. 


- 