---
layout: draft
title: "Distillation"
date: 2020-03-22 14:49:0 +0000
comments: False
share: False
categories: cv
---

**[Online Knowledge Distillation with Diverse Peers,AAAI20](https://arxiv.org/pdf/1912.00350.pdf)**

[code](https://github.com/DefangChen/OKDDip)


**[Born-Again Neural Networks,ICML18](https://arxiv.org/pdf/1805.04770.pdf)**

 We study KD from a new perspective:
rather than compressing models, we train students parameterized identically to their teachers. Surprisingly, these Born-Again Networks(BANs), outperform their teachers significantly,
both on computer vision and language modeling tasks.

check application in [Rethinking Few-Shot Image Classification: a Good Embedding Is All You Need?](https://arxiv.org/pdf/2003.11539.pdf)


**[Revisiting Knowledge Distillation via Label Smoothing Regularization,CVPR20](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Revisiting_Knowledge_Distillation_via_Label_Smoothing_Regularization_CVPR_2020_paper.pdf)**


[code](https://github.com/yuanli2333/Teacher-free-Knowledge-Distillation)

- In this work, we challenge this common belief
by following experimental observations: 1) beyond the acknowledgment that the teacher can improve the student, the
student can also enhance the teacher significantly by reversing the KD procedure; 2) a poorly-trained teacher with
much lower accuracy than the student can still improve the
latter significantly.
- we further propose a novel
Teacher-free Knowledge Distillation (Tf-KD) framework,
where a student model learns from itself or manuallydesigned regularization distribution. The Tf-KD achieves
comparable performance with normal KD from a superior teacher, which is well applied when a stronger teacher
model is unavailable. Meanwhile, Tf-KD is generic and
can be directly deployed for training deep neural networks.
Without any extra computation cost, Tf-KD achieves up
to 0.65% improvement on ImageNet over well-established
baseline models, which is superior to label smoothing regularization.



**[SSKD:Knowledge Distillation Meets Self-Supervision,ECCV20,poster](https://arxiv.org/pdf/2006.07114.pdf)**

[code](https://github.com/xuguodong03/SSKD)

-  We further
show that self-supervision signals improve conventional distillation with
substantial gains under few-shot and noisy-label scenarios.
- Given the richer knowledge mined from self-supervision, our knowledge distillation
approach achieves state-of-the-art performance on standard benchmarks,
i.e., CIFAR100 and ImageNet, under both similar-architecture and crossarchitecture settings.

## Self-Training

**[Self-training with Noisy Student improves ImageNet classification,CVPR20](https://arxiv.org/pdf/1911.04252.pdf)**

[reddit](https://www.reddit.com/r/MachineLearning/comments/dvh8e8/191104252_selftraining_with_noisy_student/)


> Our key improvements lie in adding noise to the student
and using student models that are equal to or larger than the
teacher. This makes our method different from Knowledge
Distillation [33] where adding noise is not the core concern
and a small model is often used as a student to be faster than
the teacher. One can think of our method as Knowledge
Expansion in which we want the student to be better than
the teacher by giving the student model enough capacity and
difficult environments in terms of noise to learn through.

**[Rethinking Pre-training and Self-training,Arxiv2006](https://arxiv.org/pdf/2006.06882.pdf)**

[zhihu](https://www.zhihu.com/question/401621721)

Our study reveals the generality and
flexibility of self-training with three additional insights: 

- 1) stronger data augmentation and more labeled data further diminish the value of pre-training, 
- 2) unlike pre-training, self-training is always helpful when using stronger data augmentation,
in both low-data and high-data regimes
- 3) in the case that pre-training is helpful, self-training improves upon pre-training.


**[Improving Semantic Segmentation via Self-Training,Arxiv2004](https://arxiv.org/pdf/2004.14960.pdf)**

![](/imgs/self-training-seg.png)





