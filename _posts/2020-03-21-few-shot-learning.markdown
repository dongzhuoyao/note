---
layout: draft
title: "Few-shot learning"
date: 2020-03-22 14:49:0 +0000
comments: False
share: False
categories: cv
---


benchmark(https://few-shot.yyliu.net/miniimagenet.html)

## Few-shot learning

check [Meta-Learning in Neural Networks: A Survey](https://arxiv.org/pdf/2004.05439.pdf)

**[Matching Networks for One Shot Learning,NIPS16](https://arxiv.org/pdf/1606.04080.pdf)**

**[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks,ICML17](https://arxiv.org/pdf/1703.03400.pdf)**

**[One-Shot Generalization in Deep Generative Model,JMLR16](https://arxiv.org/pdf/1603.05106.pdf)**


**[Bayesian Few-Shot Classification with One-vs-Each Pólya-Gamma Augmented Gaussian Processes,Arxiv2007](https://arxiv.org/pdf/2007.10417.pdf)**

**[CrossTransformers: spatially-aware few-shot transfer,Arxiv2007](https://arxiv.org/pdf/2007.11498.pdf)**

 - In this work, we illustrate how the neural network
representations which underpin modern vision systems are subject to supervision
collapse, whereby they lose any information that is not necessary for performing
the training task, including information that may be necessary for transfer to new
tasks or domains. We then propose two methods to mitigate this problem. First, we
employ self-supervised learning to encourage general-purpose features that transfer
better. Second, we propose a novel Transformer based neural network architecture
called CrossTransformers, which can take a small number of labeled images and
an unlabeled query, find coarse spatial correspondence between the query and the
labeled images, and then infer class membership by computing distances between
spatially-corresponding features.
- state-of-the-art performance on Meta-Dataset, a recent dataset for evaluating transfer from ImageNet to many other
vision datasets.



**[DeepEMD: Few-Shot Image Classification with Differentiable Earth Mover's Distance and Structured Classifiers,CVPR20,oral](https://arxiv.org/abs/2003.06777)**

need to setup the formulation of cost c, supply s and demand d in advance. c is decided in cosine similary distance between patch features. s and d is setup by cross-reference mechanism(session4.4 )

Intuition of setuping s and d: Intuitively, the node with a larger weight plays a more important role in the comparison of two sets, while a node with a very small weight can hardly influence the overall distance no matter which nodes it matches with.


**[Adaptive Cross-Modal Few-shot Learning,NIPS19](https://arxiv.org/pdf/1902.07104.pdf)**

new task

**[A New Meta-Baseline for Few-Shot Learning,Arxiv2003](https://arxiv.org/abs/2003.04390)**

[code](https://github.com/cyvius96/few-shot-meta-baseline)

![](/imgs/fsl2003.png)

**[RFS:Rethinking Few-Shot Image Classification: a Good Embedding Is All You Need?,Arxiv2003](https://arxiv.org/pdf/2003.11539.pdf)**

[code](https://github.com/WangYueFt/rfs/)


- miniImageNet, tieredImageNet, CIFAR-FS, and FC100, Meta-Dataset

**[Improving Few-Shot Learning using Composite Rotation based Auxiliary Task,Arxiv2006](https://arxiv.org/pdf/2006.15919.pdf)**

Based on RFS.

**[Self-Supervised Learning For Few-Shot Image Classification,Arxiv1911](https://arxiv.org/pdf/1911.06045.pdf)**

[code](https://github.com/phecy/SSL-FEW-SHOT)


Mini80-SSL is self-supervised trained from 48,000
images (80 classes training and validation ) without labels. Mini80-
SL is supervised training using same AmdimNet by cross entropy
loss with labels. Image900-SSL is SSL trained from all images from
ImageNet1K except MiniImageNet. For CUB dataset, CUB150-
SSL is trained by SSL from 150 classes (training and validation).
CUB150-SL is the supervised trained model. Image1K-SSL is SSL
trained from all images from ImageNet1K without label


**[SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning,Arxiv1911](https://arxiv.org/pdf/1911.04623.pdf)**

Meta-iNat dataset


**[A Baseline for Few-Shot Image Classification,ICLR20](https://openreview.net/forum?id=rylXBkrYDS)**

When **fine-tuned transductively**, this outperforms
the current state-of-the-art on standard datasets such as Mini-ImageNet, TieredImageNet, CIFAR-FS and FC-100 with the same hyper-parameters.

- Dataset: ImageNet-21k, also in meta-dataset.
- The proposed approach includes a standard cross-entropy loss on the labeled support samples and a Shannon entropy loss on the unlabeled query samples.


**[LaplacianShot: Laplacian Regularized Few Shot Learning,ICML20](https://github.com/imtiazziko/LaplacianShot)**

The code is adapted from SimpleShot github.


**[Few-Shot Class-Incremental Learning via Feature Space Composition,Arxiv2006](https://arxiv.org/pdf/2006.15524.pdf)**

![](/imgs/fsl-il.png)

**[Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples,ICLR20](https://openreview.net/forum?id=rkgAGAVKPr)**

- heterogeneous dataset compared with previous homogeneous dataset.
- EFFECT OF TRAINING ON ALL DATASETS OVER TRAINING ON ILSVRC-2012 ONLY:As discussed in the main paper, we notice that we do not always observe a clear generalization advantage in training from a wider collection of image datasets.

[iclr forum](https://openreview.net/forum?id=rkgAGAVKPr)

[code only tf](https://github.com/google-research/meta-dataset)

**[Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation,ICLR20](https://openreview.net/forum?id=SJl5Np4tPr)**

still need to see the out-of-domain images when training. similar to transductive setting.[check code](https://github.com/hytseng0509/CrossDomainFewShot/blob/master/methods/LFTNet.py#L99)

- auxiliary classifier: The training in the initial stage is not stable and may harm the model performance. We use the auxiliary training to solve the problem and decay the weight of the auxiliary training loss for later epochs.
- The learned transformation layers are not used when testing. The feature-wise transformation layers are used only during the training phase to improve the model generalization.
- gnn implementation: The code we provide here can only train the model with the pre-trained feature encoder. You can refer to the original implementation here for training the model from scratch.


**[Few-shot Classification via Adaptive Attention,Arxiv2008](https://arxiv.org/pdf/2008.02465.pdf)**


**[Associative Alignment for Few-shot Image Classification,Arxiv1912](https://arxiv.org/pdf/1912.05094.pdf)**

This paper proposes
the idea of associative alignment for leveraging part of the base data by
aligning the novel training instances to the closely related ones in the
base training set. This expands the size of the effective novel training set
by adding extra “related base” instances to the few novel ones, thereby
allowing a constructive fine-tuning.

**[A Broader Study of Cross-Domain Few-Shot Learning,ECCV20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720120.pdf)**

focus on aerial and medical imaging.

[code](https://github.com/IBM/cdfsl-benchmark)

[related workshop](https://www.learning-with-limited-labels.com/challenge)

![](/imgs/bscd-fsl.png)

**[EPNet:Embedding Propagation: Smoother Manifold for Few-Shot Classification,ECCV20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710120.pdf)**

b.......

**[SEN: A Novel Feature Normalization Dissimilarity Measure for Prototypical Few-Shot Learning Networks,ECCV20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680120.pdf)**

no code.

**[Impact of base dataset design on few-shot image classification,ECCV20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610579.pdf)**

What is the influence of the similarity between base and test classes?
Given a fixed annotation budget, what is the optimal trade-off between
the number of images per class and the number of classes? Given a fixed
dataset, can features be improved by splitting or combining different
classes? Should simple or diverse classes be annotated?


**[TAFSSL: Task-Adaptive Feature Sub-Space Learning for few-shot classification,ECCV20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520511.pdf)**

While number of techniques have been proposed for FSL, several factors
have emerged as most important for FSL performance, awarding SOTA
even to the simplest of techniques. These are: the backbone architecture
(bigger is better), type of pre-training (meta-training vs multi-class),
quantity and diversity of the base classes (the more the merrier), and using auxiliary self-supervised tasks (a proxy for increasing the diversity).
In this paper we propose TAFSSL, a simple technique for improving
the few shot performance in cases when some additional unlabeled data
accompanies the few-shot task


## Few-shot segmentation


**[Prototype Mixture Models for Few-shot Semantic Segmentation,ECCV20](https://arxiv.org/pdf/2008.03898.pdf)**

[code](https://github.com/Yang-Bob/PMMs)

Our approach utilizes CANet without iterative
optimization as the baseline, which uses VGG16 or ResNet50 as backbone CNN
for feature extraction.



**[Prior Guided Feature Enrichment Network for Few-Shot Segmentation,Arxiv2008](https://arxiv.org/pdf/2008.01449.pdf)**


**[On the Texture Bias for Few-Shot CNN Segmentation,Arxiv2003](https://arxiv.org/pdf/2003.04052.pdf)**


**[CRNet: Cross-Reference Networks for Few-Shot Segmentation,CVPR20](https://arxiv.org/pdf/2003.10658.pdf)**

The motivation is interesting, k-pairs  can be utilized as k2 times. 

The design of cross-reference includes a elementwise multiplication of two sigmoids. the intuition behind is , only the common features in the two branches will have a high activation in
the fused importance vector. Finally, we use the fused vector to weight the input feature maps to generate reinforced feature representations.

The condition module is quite simple by upsampling+concatenation along channel dimension.

Also the thought of refinement is also worthy of learning.


**[Self-Supervised Tuning for Few-Shot Segmentation,Arxiv2004](https://arxiv.org/pdf/2004.05538.pdf)**

**[Pyramid Graph Networks with Connection Attentions for Region-Based
One-Shot Semantic Segmentation,ICCV19](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Pyramid_Graph_Networks_With_Connection_Attentions_for_Region-Based_One-Shot_Semantic_ICCV_2019_paper.pdf)**


**[One-Shot Segmentation in Clutter,ICML18](https://arxiv.org/pdf/1803.09597.pdf)**

**[CANet: Class-Agnostic Segmentation Networks with Iterative Refinement and
Attentive Few-Shot Learning,CVPR19](https://arxiv.org/pdf/1903.02351.pdf)**

Check Fig 2.

**[PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment,ICCV19](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_PANet_Few-Shot_Image_Semantic_Segmentation_With_Prototype_Alignment_ICCV_2019_paper.pdf)**

Masked Average Pooling + Cosine similarity, obtain final segmentation result.

**[Objectness-Aware One-Shot Semantic Segmentations,Arxiv20,April](https://arxiv.org/pdf/2004.02945.pdf)**

- adopt HRNetV2-W48 as the backbone of the objectness module. 
- The objectness module is trained to segment out all objects in the image.(train the objectness module for 300,000 iterations with batch size 4, which takes about 50 hours on GeForce GTX 1080Ti. )
- Check Fig 2, support feature, query feature, and objectness feature are congregated by adding operation.


**[Weakly Supervised Few-shot Object Segmentation using Co-Attention with Visual and Semantic Inputs,Arxiv20](https://arxiv.org/pdf/2001.09540.pdf)**

only requiring image-level classification data for few-shot object segmentation. propose a novel multi-modal interaction module for few-shot object segmentation that utilizes a coattention mechanism using both visual and word embedding.

Class wording embedding is then spatially tiled and concatenated with the visual features resulting in flattened matrix representations. 

Unlike non-local block relating $$WH \times C$$ and $$C \times WH$$, they add an extra $$C \times C$$ matrix in the very middle. Also, they consider two-directions by applying softmax along different dimensions. Check Fig 2.

**[Attention-based Multi-Context Guiding for Few-Shot Semantic Segmentation,AAAI19](http://taohu.me/pdf/few-shot-seg.pdf)**

**[Part-aware Prototype Network for Few-shot Semantic Segmentation,ECCV20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540137.pdf)**

.....

[code](https://github.com/Xiangyi1996/PPNet-PyTorch)

**[Few-Shot Semantic Segmentation with Democratic Attention Networks,ECCV20](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580715.pdf)**

no code





## Few-shot detection


**[UFO2: A Unified Framework towards Omni-supervised Object Detection,ECCV20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640290.pdf)**

no code

UFO2 incorporates strong supervision (e.g., boxes), various
forms of partial supervision (e.g., class tags, points, and scribbles), and
unlabeled data. Through rigorous evaluations, we demonstrate that each
form of label can be utilized to either train a model from scratch or to
further improve a pre-trained model. 

![](/imgs/UFO2.png)

How to uniform?

**[OS2D: One-Stage One-Shot Object Detection by Matching Anchor Features(ECCV20)](https://arxiv.org/pdf/2003.06800.pdf)**


**[Multi-Scale Positive Sample Refinement for Few-Shot Object Detection,ECCV20](https://arxiv.org/abs/2007.09384)**

competitor: YOLO-FS,meta-RCNN;dataset: VOC,COCO.

[code](https://github.com/jiaxi-wu/MPSR/issues)




**[Meta-RCNN: Meta Learning for Few-Shot Object Detection,ICLR20, reject](https://openreview.net/forum?id=B1xmOgrFPS)**

bbbo


**[Meta r-cnn: Towards general solver for instance-level low-shot learning,ICCV19](https://yanxp.github.io/metarcnn.html)**


[code](https://github.com/yanxp/MetaR-CNN)

**[RepMet: Representative-based metric learning for classification and few-shot object detection,CVPR19](https://github.com/jshtok/RepMet)**

**[Meta-Learning to Detect Rare Objects,ICCV19](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Meta-Learning_to_Detect_Rare_Objects_ICCV_2019_paper.pdf)**

**[Few-Shot Object Detection with Attention-RPN and Multi-Relation Detector,CVPR20](https://arxiv.org/abs/1908.01998)**

**[Context-Transformer: Tackling Object Confusion for Few-Shot Detection,AAAI20](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-YangZ.2509.pdf)**

**[Weakly-supervised Any-shot Object Detection,Arxiv2006](https://arxiv.org/abs/2006.07502)**

**[Frustratingly Simple Few-Shot Object Detection,ICML20](https://arxiv.org/pdf/2003.06957.pdf)**

[code](https://github.com/ucbdrive/few-shot-object-detection)

new benchmarks on PASCAL VOC, COCO and LVIS.

![](/imgs/frustrating-fsd.png)

**[Few-Shot Object Detection and Viewpoint Estimation for Objects in the Wild,ECCV20](https://arxiv.org/pdf/2007.12107.pdf)**

[code](http://imagine.enpc.fr/~xiaoy/FSDetView/)


## Few-shot instance segmentation

[One-Shot Instance Segmentation,Arxiv](https://arxiv.org/pdf/1811.11507.pdf)

[Differentiable Meta-learning Model for Few-shot Semantic Segmentation,AAAI20](https://arxiv.org/pdf/1911.10371.pdf)

[FGN: Fully Guided Network for Few-Shot Instance Segmentation,CVPR20](https://arxiv.org/abs/2003.13954)


## Few-shot video classification

**[TAEN: Temporal Aware Embedding Network for Few-Shot Action Recognition,Arxiv2004](https://arxiv.org/pdf/2004.10141.pdf)**


**[Generalized Many-Way Few-Shot Video Classification,Arxiv2007](https://arxiv.org/pdf/2007.04755.pdf)**

In this work, we point out that a spatiotemporal CNN trained on a large-scale video
dataset saturates existing few-shot video classification benchmarks. Hence, we propose
new more challenging experimental settings, namely generalized few-shot video classification (GFSV) and few-shot video classification with more ways than the classical
5-way setting. We further improve spatiotemporal CNNs by leveraging the weaklylabeled videos from YFCC100M using weak-labels such as tags for text-supported and
video-based retrieval. Our results show that generalized more-way few-shot video classification is challenging and we encourage future research in this setting

**[Few-shot Action Recognition with Permutation-invariant Attention,ECCV20,splotlight](https://arxiv.org/pdf/2001.03905.pdf)**





## Few-shot 3D cloud

**[Few-shot 3D Point Cloud Semantic Segmentation,Arxiv2006](https://arxiv.org/pdf/2006.12052.pdf)**


## Few-shot Edge Detection

[CAFENet: Class-Agnostic Few-Shot
Edge Detection Network,Arxiv](https://arxiv.org/pdf/2003.08235.pdf)

## Few-shot video activity localization 

**[METAL: Minimum Effort Temporal Activity Localization in Untrimmed Videos,CVPR20](https://sites.cs.ucsb.edu/~yfwang/papers/cvpr2020.pdf)**

**[TAEN: Temporal Aware Embedding Network for Few-Shot Action Recognition,Arxiv2004](https://arxiv.org/pdf/2004.10141.pdf)**

**[Weakly-Supervised Video Re-Localization with Multiscale Attention Model,AAAI20](http://vllab.cs.nctu.edu.tw/images/paper/aaai-huang20.pdf)**


## imbalanced classification

## Long-tailed recognition

Datasets exhibit a natural power law distribution[^devilinthedetails], allowing us to assess
model performance on four folds, Manyshot classes (≥ 100 samples), Mediumshot
classes (20 ∼ 100 samples), Fewshot classes (< 20 samples), and All classes. 


Typical dataset, LVIS, etc.

**typical solution**

loss reweighting, data re-sampling, or transfer learning from head- to tail-classes.  For details can check 
[Decoupling Representation and Classifier for Long-Tailed Recognition,ICLR20](https://openreview.net/forum?id=r1gRTCVFvB).


- One of the commonly used methods in re-sampling is oversampling, which randomly samples more training data from the minority classes, to tackle the unbalanced class distributionClass-aware sampling, also called class-balanced sampling, is a typical technique of oversampling, which first samples a category and then an image uniformly that contains the sampled category.
- While oversampling methods achieve significant improvement for under-represented classes, they come with a high potential risk of overfitting. 
- On the opposite of oversampling, the main idea of under-sampling is to remove some available data from frequent classes to make the data distribution more balanced. However, the under-sampling is infeasible in extreme long-tailed datasets, since the imbalance ratio between the head class and tail class are extremely large.

**typical dataset**: ImageNet-LT(Liu et al.), iNaturalist 2018, Places-LT(Liu et al.),CIFAR-LT(created by  Cui et al.)

**[Long-Tailed Recognition Using Class-Balanced Experts,Arxiv2004](https://arxiv.org/pdf/2004.03706.pdf)**




**[Decoupling Representation and Classifier for Long-Tailed Recognition,ICLR20](https://openreview.net/forum?id=r1gRTCVFvB)**

decouple the learning procedure into representation learning and classification, with
representations learned with the simplest instance-balanced (natural) sampling, it
is also possible to achieve strong long-tailed recognition ability at little to no cost
by adjusting only the classifier.

Reviewer's opinion.


> In general, this is paper is an interesting paper. The author propose that instance-balanced sampling already learns the best and most generalizable representations, which is out of common expectation. They perform extensive experiment to illustrate their points.

> We further investigate if we can **automatically learn the tau value instead of grid search**. To this end, following cRT, we set tau as a learnable parameter and learn it on the training set with balanced sampling,  while keeping all the other parameters fixed (including both backbone network and classifier). Also, we compare the learned tau value and the corresponding results in the above table (denoted by ‘learn’ = ‘Y’).  This further reduces the manual effort of searching best tau values and make the strategy more accessible for practical usage. We will incorporate these new findings in the paper, and once again, we thank all reviewers for the inspiring comments. All above discussion is updated to our manuscript in Appendix B.5. 



**[Learning to Segment the Tail,CVPR20](https://arxiv.org/abs/2004.00900)**

**[BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition,CVPR20,oral](https://github.com/Megvii-Nanjing/BBN)**

same intuition as  ICLR20 paper above. check Fig 2.


**[Equalization Loss for Long-Tailed Object Recognition,CVPR20](https://arxiv.org/pdf/2003.05176.pdf)**

check [zhihu](https://www.zhihu.com/question/372070853/answer/1082980270), mainly focused on detection task in [LVIS](https://www.lvisdataset.org/challenge).check short report [here](https://www.lvisdataset.org/assets/challenge_reports/2019/strangeturtle_equalization_loss.pdf).


**[Large-Scale Long-Tailed Recognition in an Open World,CVPR19,oral](https://liuziwei7.github.io/projects/LongTail.html)**

define Open Long-Tailed Recognition (OLTR) as learning from such naturally distributed data and optimizing the classification accuracy over a balanced test set which include head, tail, and open classes.

The work fills the void in practical
benchmarks for imbalanced classification, few-shot learning, and open-set recognition, enabling future research that
is directly transferable to real-world applications.

 develop an integrated OLTR algorithm that maps
an image to a feature space such that visual concepts can
easily relate to each other based on a learned metric that respects the closed-world classification while acknowledging
the novelty of the open world.

**[Class-Balanced Loss Based on Effective Number of Samples,CVPR19](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8953804&tag=1)**

- Effective number introduced, check Fig 1 for intuition. "Re-weighted by effective number of samples is better than reweighting by inverse class frequency"
- Check Fig 3 for visual understading of effective number.
- design a re-weighting scheme that uses the effective number of samples for each class to re-balance the loss, thereby yielding a class-balanced loss
- Comprehensive experiments are conducted on artificially induced long-tailed CIFAR datasets and large-scale datasets including ImageNet and iNaturalist. Our results show that when trained with the proposed class-balanced loss, the network is able to achieve significant performance gains on long-tailed datasets.

**[Trainable Undersampling for Class-Imbalance Learning,AAAI19](https://www.semanticscholar.org/paper/Trainable-Undersampling-for-Class-Imbalance-Peng-Zhang/d349207dee9dd782c34a6a6cd6d71fd5eb178d3a)**







## TTT

**[Test-Time Training with Self-Supervision for Generalization under Distribution Shifts,ICML20](https://arxiv.org/pdf/1909.13231.pdf)**

**[Self-Supervised Policy Adaptation during Deployment,Arxiv2007](https://nicklashansen.github.io/PAD/)**

**[Fully Test-time Adaptation by Entropy Minimization,Arxiv2006](https://arxiv.org/abs/2006.10726)**













#### Footnotes
* footnotes will be placed here. This line is necessary
{:footnotes}

[^devilinthedetails]: The devil is in the tails: Fine-grained classification in the wild.






