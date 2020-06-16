---
layout: draft
title: "Adversarial Attack"
permalink: /adversarial_attack
date: 2020-03-31 14:49:0 +0000
comments: False
share: False
categories: cv
---

check [Adversarial Attacks and Defenses in Deep Learning](https://reader.elsevier.com/reader/sd/pii/S209580991930503X?token=0E2082450252D01B31369E260C0B8DA7DABDEE244A0F01D7C561B94C23C8A78F2F03E8F7D3604CDE0F7934A30CE90B29)



**[L-BFGS]()**

$$
min_{x} ||x - x'||_{p} \text{subject to } f(x')\neq y'
$$

**[Fast Gradient Sign,FGSM,ICLR15,highly cited](https://arxiv.org/abs/1412.6572)**

$$
x' = x + \epsilon\cdot sign[\Delta_{x} J(\theta,x,y)]
$$

it can be easily changed to a targeted attack

$$
x' = x - \epsilon\cdot sign[\Delta_{x} J(\theta,x,y')]
$$

The fact that these simple, cheap algorithms are able to generate misclassified examples serves as
evidence in favor of our interpretation of adversarial examples as a result of linearity. 

**[Basic Iterative Method (BIM)](https://arxiv.org/abs/1607.02533)**

improve the performance ofFGSM by running a finer iterative optimizer for multiple iterations.The BIM performs FGSM with a smaller step size and clips theupdated adversarial sample into a valid range forTiterations; thatis, in thetth iteration, the update rule is the following:

$$
x'_{t+1} = \text{Clip}(x'_{t} + \alpha \cdot \text{sign} [\Delta_{x} J(\theta, x'_{t},y)])
$$

this is a special case of PGD:

$$
x'_{t+1} = \text{Proj}(x'_{t} + \alpha \cdot \text{sign} [\Delta_{x} J(\theta, x'_{t},y)])
$$