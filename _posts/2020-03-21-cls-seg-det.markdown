---
layout: draft
title: "Classification, Segmentation, Detection"
permalink: /cls_seg_det
date: 2020-03-22 14:49:0 +0000
comments: False
share: False
categories: cv
---

## Detection

For main progress check [Survey 2019](https://link.springer.com/content/pdf/10.1007/s11263-019-01247-4.pdf)

#### Mainstream progress

**[End-to-End Object Detection with Transformers,Arxiv2005](https://arxiv.org/pdf/2005.12872.pdf)**

**[EfficientDet: Scalable and Efficient Object Detection,CVPR20](https://arxiv.org/pdf/1911.09070.pdf)**

**[Dynamic R-CNN: Towards High Quality Object Detection via Dynamic Training,Arxiv2004](https://arxiv.org/pdf/2004.06002.pdf)**

Dynamic R-CNN to adjust the label assignment criteria (IoU threshold) and the shape of regression loss function (parameters of SmoothL1 Loss) automatically based on the statistics of proposals during training

**[YOLOv4](https://arxiv.org/pdf/2004.10934.pdf)**

Creating a CNN that operates in real-time on a conventional GPU, and for which
training requires only one conventional GPU.


Need a careful check for practical tricks in det task.

[zhihu](https://www.zhihu.com/question/390191723/answer/1177584901)

- Mosaic
- Self-adversarial training
- Cross mini-batch normalization
- Pointwise SAM


#### Anchor-free

**[FCOS: A Simple and Strong Anchor-free Object Detector](https://arxiv.org/pdf/2006.09214.pdf)**


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
  
**[Stitcher: Feedback-driven Data Provider for Object Detection,Arxiv2004](https://arxiv.org/pdf/2004.12432.pdf)**

![](/imgs/stitcher.png)

[zhihu](https://www.zhihu.com/question/390191723/answer/1185984775)

similar to Mosaic tricks in YOLOv4

feedback-driven data provider is interesting




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
  



## Classification


**[ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness,ICLR19,oral](https://openreview.net/forum?id=Bygh9j09KX)**

[zh](https://zhuanlan.zhihu.com/p/61116317)


**[Spatially Attentive Output Layer for Image Classification,CVPR20](https://arxiv.org/pdf/2004.07570.pdf)**

waiting for code.



**[NetVLAD: CNN architecture for weakly supervised place recognition,CVPR16](https://arxiv.org/pdf/1511.07247.pdf)**


**[Adversarial Examples Improve Image Recognition,CVPR20](https://arxiv.org/pdf/1911.09665.pdf)**

 - propose to use two batch norm statistics, one for clean images and one auxiliary for adversarial examples. The two batchnorms properly disentangle the two distributions at normalization layers for accurate statistics estimation. We show this distribution disentangling is crucial, enabling us to successfully improve, rather than degrade, model performance with adversarial examples
- the first to show adversarial examples can improve model performance in the fully-supervised setting on the large-scale ImageNet dataset.
- a simple  auxiliary BN design, check Fig 3.

#### Architecture

**[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,ICML19](https://arxiv.org/pdf/1905.11946.pdf)**

**[HRNet,PAMI20](https://arxiv.org/pdf/1908.07919.pdf)**

check Fig 2.

**[Res2Net,PAMI20](https://arxiv.org/pdf/1904.01169.pdf)**

The Res2Net strategy exposes a new dimension, namely scale
(the number of feature groups in the Res2Net block), as an
essential factor in addition to existing dimensions of depth,
width, and cardinality. 

**[DHM,CVPR20](https://arxiv.org/pdf/2003.10739.pdf)**


**[Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution,ICCV19](https://arxiv.org/pdf/1904.05049.pdf)**

[reddit discussion](https://www.reddit.com/r/MachineLearning/comments/bdn5ix/190405049_drop_an_octave_reducing_spatial/)

**[Multigrid Neural Architectures,CVPR17](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ke_Multigrid_Neural_Architectures_CVPR_2017_paper.pdf)**
![](/imgs/mgconv.png)




#### Visualization-related

CAM

[grad-CAM](https://arxiv.org/abs/1610.02391)

## Semantic Segmentation

Check the survey [here](https://arxiv.org/pdf/2001.05566.pdf).



**[CascadePSP: Toward Class-Agnostic and Very High-Resolution Segmentation via Global and Local Refinement,CVPR20](https://github.com/hkchengrex/CascadePSP)**

**[CFNet:Co-occurrent Features in Semantic Segmentation,CVPR19](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Co-Occurrent_Features_in_Semantic_Segmentation_CVPR_2019_paper.pdf)**


similar to non-local block

**[PSANet,ECCV18](https://hszhao.github.io/papers/eccv18_psanet.pdf)**

**[DANet](https://arxiv.org/pdf/1809.02983.pdf)**

non-local on channel and spatial.


**[Context Prior,CVPR20](https://arxiv.org/pdf/2004.01547.pdf)**
Learn a $$WH \times WH$$ affinity matrix,
k=11 in context aggregation is vital for the functionality of Context Prior, without k=11, CP cannot work.

Affinity matrix construction is similar to **[Adaptive Pyramid Context Network for Semantic Segmentation,CVPR19](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Adaptive_Pyramid_Context_Network_for_Semantic_Segmentation_CVPR_2019_paper.pdf)**



**[Class-wise Dynamic Graph Convolution for Semantic Segmentation,ECCV20](https://arxiv.org/pdf/2007.09690.pdf)**


## Instance Segmentation

**[Deep Snake for Real-Time Instance Segmentation,CVPR20,oral](https://arxiv.org/pdf/2001.01629.pdf)**

<!--https://www.zhihu.com/question/360594484-->

**[SOLOv2](https://arxiv.org/pdf/2003.10152.pdf)**



## Human-object interaction

[Spatial Priming for Detecting Human-Object Interactions,Arxiv2004](https://arxiv.org/pdf/2004.04851.pdf)






