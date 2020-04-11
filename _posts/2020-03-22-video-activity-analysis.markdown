---
layout: draft
title: "Video Activity Analysis"
date: 2020-03-22 14:49:0 +0000
comments: False
share: False
categories: cv
---

## Acctivity classification

**[TEA: Temporal Excitation and Aggregation for Action Recognition,CVPR20](https://arxiv.org/pdf/2004.01398.pdf)**

- the FLOPs of 3D CNN methods is much larger than 2D CNN methods. The superiority of our TEA on Something-Something is quite impressive. Check Table 2.
- motion excitation module is inspired by SENet, Mutiple temporal aggregation(MTA) is inspired by Res2Net.
- achieves impressive results at low FLOPs on several action recognition benchmarks, such as Kinetics Something-Something, HMDB51, and UCF101.



**[Temporal Pyramid Network for Action Recognition,CVPR20](https://arxiv.org/pdf/2004.03548.pdf)**

dataset: Kinetics-400, Something-Something, Epic-Kitchen. 8GPU.

## Spatial-tempotal localization

**[Actions as Moving Points(MOC-detector)](https://arxiv.org/pdf/2001.04608.pdf)**

Anchor-free based activity localization.

Check Fig 2. An action tubelet is represented by its center point in the key frame and offsets of other frames with respect to this center point. To determine the tubelet shape, we directly regress the bounding box size along the moving point trajectory on each frame.

