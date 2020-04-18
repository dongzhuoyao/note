---
layout: draft
title: "Few-shot learning"
date: 2020-03-22 14:49:0 +0000
comments: False
share: False
categories: cv
---

## Few-shot learning

check [Meta-Learning in Neural Networks: A Survey](https://arxiv.org/pdf/2004.05439.pdf)

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

## Few-shot segmentation

**[Self-Supervised Tuning for Few-Shot Segmentation,Arxiv2004](https://arxiv.org/pdf/2004.05538.pdf)**

**[Pyramid Graph Networks with Connection Attentions for Region-Based
One-Shot Semantic Segmentation,ICCV19](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Pyramid_Graph_Networks_With_Connection_Attentions_for_Region-Based_One-Shot_Semantic_ICCV_2019_paper.pdf)**


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


**[Attention-based Multi-Context Guiding for Few-Shot Semantic Segmentation](http://taohu.me/pdf/few-shot-seg.pdf)**

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


#### Footnotes
* footnotes will be placed here. This line is necessary
{:footnotes}

[^devilinthedetails]: The devil is in the tails: Fine-grained classification in the wild.






