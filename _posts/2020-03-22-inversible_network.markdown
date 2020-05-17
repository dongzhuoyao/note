---
layout: draft
title: "Inversible network"
permalink: /inversible_network
date: 2020-03-22 14:49:0 +0000
comments: False
share: False
categories: ml
---

[https://zhuanlan.zhihu.com/p/73426787](https://zhuanlan.zhihu.com/p/73426787)



## Related materials

check [Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/pdf/1912.02762.pdf)

[Normalizing Flows: An Introduction and Review of Current Methods](https://arxiv.org/pdf/1908.09257.pdf)

[Invertible Models and Normalizing Flows: a retrospective (ICLR 2020 keynote slides)](https://docs.google.com/presentation/d/15RMCzCRwuKKv6fIwvGjwig2WnnP_5yzQGzcpJbq7zws/edit#slide=id.g8428c68825_0_0)

density estimation, variational inference, sample(generation) is essentially different and corelated.


## Pre history: distribution estimation

**[Gaussianization](https://papers.nips.cc/paper/1856-gaussianization.pdf)**
brief read

**[Independent Component Analysis]()**

**[Restricted Boltzmann Machines]()**

**[NADE:Neural Autoregressive Distribution Estimation,JMLR2000](https://arxiv.org/pdf/1605.02226.pdf)**

## Recent Advance

### Auto-regressive model


**[NADE:Neural Autoregressive Distribution Estimation,JMLR2000](https://arxiv.org/pdf/1605.02226.pdf)**


**[RNADE](https://arxiv.org/pdf/1306.0186.pdf)**

**[Pixel Recurrent Neural Networks,ICML16](https://arxiv.org/pdf/1601.06759.pdf)**

> Furthermore, in contrast to previous approaches that model the pixels as continuous values (e.g., Theis & Bethge (2015); Gregor et al.(2014)), we model the pixels as discrete values using a multinomial distribution implemented with a simple softmax layer.   Each channel variable xi,∗ simply takes one of 256 distinct values.

We have four types of networks: the PixelRNN based on Row LSTM, the one based on Diagonal BiLSTM, the fully convolutional one and the MultiScale one.

Have a detailed discussion about dequantizing the image data.
> In the literature it is currently best practice to add realvalued noise to the pixel values to dequantize the data when using density functions (Uria et al., 2013). When uniform noise is added (with values in the interval [0, 1]), then the log-likelihoods of continuous and discrete models are directly comparable (Theis et al., 2015). 

Evaluation details: For MNIST we report the negative log-likelihood in nats as it is common practice in literature. For CIFAR-10 and ImageNet we report negative log-likelihoods in bits per dimension. The total discrete log-likelihood is normalized by the dimensionality of the images (e.g., 32 × 32 × 3 = 3072 for CIFAR-10). These numbers are interpretable as the number of bits that a compression scheme based on this model would need to compress every RGB color value (van den Oord & Schrauwen, 2014b; Theis et al., 2015); in practice there is also a small overhead due to arithmetic coding.



**[PixelCNN:Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/pdf/1606.05328.pdf)**

**[Image Transformer,ICML18](https://arxiv.org/pdf/1802.05751.pdf)**

Temperature is important in generation, this point also inspires Glow:
> Across all of the presented experiments, we use categorical
sampling during decoding with a tempered softmax (Dahl
et al., 2017). We adjust the concentration of the distribution
we sample from with a temperature τ > 0 by which we
divide the logits for the channel intensities.

![](/imgs/image-transformer.png)

**[(IAF)Improved Variational Inference with Inverse Autoregressive Flow,NIPS16](https://arxiv.org/abs/1606.04934)**

[NeuIPS review](https://papers.nips.cc/paper/6581-improved-variational-inference-with-inverse-autoregressive-flow)

Preliminary: PixelCNN , PixelRNN，MADE

>  The paper are able to exploit the recent advances in autoregressive models, particularly in making efficient inference through parallel computing. However, they avoid the cumbersome sampling/inversion procedure of autoregressive model, which is quite ingenious. 

![](/imgs/iaf.png)

![](/imgs/iaf2.png)


$$
z = \sigma \odot z + (1-\sigma) \odot m
$$
is parallelized, this is the main difference betwenn autoregressive model.

Perhaps the simplest special version of IAF is one with a simple step(T=1), and a linear autoregressive
model. This transforms a Gaussian variable with diagonal covariance, to one with linear dependencies,
i.e. a Gaussian distribution with full covariance. See appendix A for an explanation.

We found that results improved when reversing the ordering of the variables after each step in the IAF
chain.

Why sampling speed is so high compared with PixelCNN?TODO

Fig 5 in supp,TODO.

**[MADE:Masked Autoencoder for Distribution Estimation,ICML15](https://arxiv.org/abs/1502.03509)**

![](/imgs/made.png)

**[MAF: Masked Autoregressive Flow for Density Estimation,NeuIPS17](https://arxiv.org/abs/1705.07057)**

Based on MADE


Difference between previous methods:

>  An early example is Gaussianization [4], which is based on successive application of independent component analysis. Enforcing invertibility with nonsingular weight matrices has been proposed [3, 29], however in such approaches calculating the determinant of the Jacobian scales cubicly with data dimensionality in general. **Planar/radial flows [27] and Inverse Autoregressive Flow (IAF) [16] are models whose Jacobian is tractable by design. However, they were developed primarily for variational inference and are not well-suited for density estimation, as they can only efficiently calculate the density of their own samples and not of externally provided datapoints.** The Non-linear Independent Components Estimator (NICE) [5] and its successor Real NVP [6] have a tractable Jacobian and are also suitable for density estimation.

Check session for detailed "Relationship with Inverse Autoregressive Flow".

> The advantage of Real NVP compared to MAF and IAF is that it
can both generate data and estimate densities with one forward pass only, whereas MAF would need
D passes to generate data and IAF would need D passes to estimate densities.

why?

Have a detail comparison beteen MADE,IAF,MAF

**[Axial Attention in Multidimensional Transformers](https://openreview.net/pdf?id=H1e5GJBtDr)**




### Normalization Flow

**[Planar and Radial Flows]()**

Planar Flow:
$$
g(x) = x + u h(w^{T}x +b)
$$

An extension based on Planar flow is Sylvester flow:
$$
g(x) = x + U h(W^{T}x +b)
$$
where U and W are DxM matrices

Radial Flows:

$$
g(x) = x + \frac{\beta}{\alpha + ||x-x_{0}||}(x-x_{0})
$$


**[NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION,ICLRW15](https://arxiv.org/abs/1410.8516)**

core idea of coupling layer(actually also proposed a genneral coupling layer, while they use additive coupling layer for simplicity.):

$$
y_{1} =x_{1}\\
y_{2} = x_{2} + m(x_{1})
$$

m can be as complex as you need, I like this idea, why the fucking ICLR reject it? Also this paper is honest compared with common papers. show the simple intuition in the very beginning.

> Examining the Jacobian, we observe that at least
three coupling layers are necessary to allow all dimensions to influence one another. We generally use four.

Prior distribution can be gaussian distribution or logistic distribution. Their prior distribution can be explicitly expressed in session3.4(EXCERCISE)

Difference between VAE: Like the variational auto-encoders, the NICE model uses an encoder to avoid the difficulties of inference, but its encoding is deterministic. The log-likelihood is tractable and the training procedure does not require any sampling (apart from dequantizing the data).

SCALING intuition: As each additive coupling layers has unit Jacobian determinant (i.e. is volume preserving), their composition will necessarily have unit Jacobian determinant too.(TODO)This allows the learner to give more weight (i.e. model more variation) on some dimensions and less in others. similar to attention mechanism recently. 

The INPAINTING application is interesting, a super simple projected gradient ascent is applied based on the pre-trained combination probability between H and O. 

The change of variable formula for probability density functions is prominently used, check related works in this paper.

The NICE criterion is very similar to the criterion of the variational auto-encoder. More specifically,
as the transformation and its inverse can be seen as a perfect auto-encoder pair,... check related work.TODO

**[Density estimation using Real NVP,ICLR17](https://arxiv.org/abs/1605.08803)**


Contributions: affine coupling layer, masked convolution, multi-scale architecture(squeeze out), introduce moving-average batch normalization into this topic.

Training a normalization flow does not in theory requires a discriminator network as in GANs, or approximate inference as in variational autoencoders. If the function is bijective, it can be trained through maximum likelihood using the change of variable formula. This formula has been discussed in several papers including the maximum likelihood formulation of independent components analysis (ICA) [4, 28], gaussianization [14, 11] and deep density models [5, 50, 17, 3]. 

dive deeper into related works. TODO.

About the nature of maximum likelihood:

> As mentioned in [62, 22], maximum likelihood is a principle that values diversity over sample quality in a limited capacity setting.



**[Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/pdf/1807.03039.pdf)**


![](/imgs/glow.png)

Summairzed four merits of flow-based generative models.

ActNorm is similar to BN, without mean and standard deviation. only learn the scale and bias with size $$C\times 1\times 1$$, interesting thing is you only know how to initialize until first batch of data arrives.

An additive coupling layer proposed before is a special case with s = 1 and a log-determinant of
0 in affine coupling layers. Actually NICE also proposed a general coupling layer. So what's the difference between glow's coupling layer and the general coupling layer in NICE?

invertable 1x1 convolution by LU decomposition, TODO.

Temperature T is vital in n likelihood-based generative models.



**[RevNets:The Reversible Residual Network: Backpropagation Without Storing Activations](https://arxiv.org/pdf/1707.04585.pdf)**

intuition: present the Reversible Residual Network (RevNet), a variant of ResNets
where each layer’s activations can be reconstructed exactly from the next layer’s.
Therefore, the activations for most layers need not be stored in memory during
backpropagation.

TODO: how to map the input to a categorial outpout(softmax)? need check code.

TODO: how to do downsampling?

Note that unlike residual blocks, reversible blocks must have a stride of 1 because otherwise the layer
discards information, and therefore cannot be reversible. Standard ResNet architectures typically
have a handful of layers with a larger stride. If we define a RevNet architecture analogously, the
activations must be stored explicitly for all non-reversible layers.

Splitting is based on channel dimension.

check footnote 2 in page 4, you can feel the grid-searching is labor-consuming.

![](/imgs/revnet.png)

**[i-REVNET: DEEP INVERTIBLE NETWORKS,ICLR18](https://arxiv.org/pdf/1802.07088.pdf)**

smart idea: It is widely believed that the success of deep convolutional networks is based on
progressively discarding uninformative variability about the input with respect to
the problem at hand. This is supported empirically by the difficulty of recovering
images from their hidden representations, in most commonly used network architectures. In this paper we show via a one-to-one mapping that this loss of information is not a necessary condition to learn representations that generalize well on complicated problems, such as ImageNet.

The design is similar to the Feistel cipher diagrams (Menezes et al., 1996) or a lifting scheme (Sweldens, 1998), which are invertible and efficient implementations of complex transforms like second generation wavelets.

In this way, we avoid the non-invertible modules of a RevNet (e.g. max-pooling or strides) which
are necessary to train them in a reasonable time and are designed to build invariance w.r.t. translation
variability.

The method part is too abstract to understand, need more time to figure it out, TODO.


**[Set Flow: A Permutation Invariant Normalizing Flow,Arxiv1909](https://arxiv.org/pdf/1909.02775.pdf)**


**[Multi-variate Probabilistic Time Series Forecasting via Conditioned Normalizing Flows,Arxiv2002](https://arxiv.org/pdf/2002.06103.pdf)**

**[Graph Normalizing Flows,Arxiv1905](https://arxiv.org/pdf/1905.13177.pdf)**

**[Your classifier is secretly an energy based model and you should treat it like one,ICLR20,oral](https://openreview.net/forum?id=Hkxzx0NtDB)**

> This paper advocates the use of energy based models (EBMs) to help realize the potential of generative models on downstream discriminative problems.

**[Variational autoencoders and nonlinear ICA: A unifying framework,AISTAT20](https://arxiv.org/abs/1907.04809)**

Preliminary work: [Nonlinear ICA Using Auxiliary Variables,AISTAT19](https://arxiv.org/pdf/1805.08651.pdf)

check conclusion part.

> The framework of variational autoencoders allows us to efficiently learn deep latent-variable
models, such that the model’s marginal distribution over observed variables fits the data.
Often, we’re interested in going a step further,
and want to approximate the true joint distribution over observed and latent variables,
including the true prior p(z) and posterior p(z|x) distributions over latent variables. This is known
to be generally impossible due to unidentifiability of the model.

The VAE model actually learns a full generative model
$$p_{\theta}(x,z) = p_{\theta}(x|z)p_{\theta}(z)
$$ 
and an inference model $$q_{\theta}(z|x)$$ that approximates its posterior $$p_{\theta}(z|x)$$ The problem is
that we generally have no guarantees about what these
learned distributions actually are: all we know is that
the marginal distribution over x is meaningful (Eq. 3).
The rest of the learned distributions are, generally,
quite meaningless.

> Almost no literature exists on achieving this goal. A
pocket of the VAE literature works towards the related
goal of disentanglement, but offers no proofs or theoretic guarantees of identifiability of the model or its latent variables.


The prior on the latent variables pθ(z|u) is assumed
to be conditionally factorial, where each element of
zi ∈ z has a univariate exponential family distribution
given conditioning variable u. To this end, in
practice we choose the prior pθ(z|u) to be a Gaussian
location-scale family, which is widely used with VAE.





**[Disentanglement by Nonlinear ICA with General Incompressible-flow Networks (GIN),ICLR20](https://openreview.net/forum?id=rygeHgSFDH)**

repeat

**[Categorical Reparameterization with Gumbel-Softmax,ICLR17](https://arxiv.org/abs/1611.01144)**


**[Gaussianization Flows,Arxiv2003](https://arxiv.org/pdf/2003.01941.pdf)**


**[Flow Contrastive Estimation of Energy-Based Models,Arxiv1912](https://arxiv.org/pdf/1912.00589.pdf)**

**[ICE-BeeM: Identifiable Conditional Energy-Based Deep Models,Arxiv2002](https://arxiv.org/pdf/2002.11537.pdf)**

**[Analyzing Inverse Problems with Invertible Neural Networks,ICLR19](https://openreview.net/forum?id=rJed6j0cKX)**




**[Sylvester Normalizing Flows for Variational Inference,UAI18](https://arxiv.org/abs/1803.05649)**


**[BayesFlow: Learning complex stochastic models with invertible neural networks,Arxiv2003](https://arxiv.org/abs/2003.06281)**



**[Do Deep Generative Models Know What They Don't Know?,ICLR19](https://arxiv.org/pdf/1810.09136.pdf)**

**[FFJORD: Free-Form Continuous Dynamics for Scalable Reversible Generative Models,ICLR19](https://openreview.net/forum?id=rJxgknCcK7)**

**[PointFlow: 3D Point Cloud Generation with Continuous Normalizing Flows,ICCV19](https://arxiv.org/abs/1906.12320)**

**[C-Flow: Conditional Generative Flow Models for Images and 3D Point Clouds,ICCV19]()**

**[Hybrid Models with Deep and Invertible Features,ICML19](https://arxiv.org/pdf/1902.02767.pdf)**

> We are unaware of any work that uses normalizing flows as the generative component of a hybrid model. The most related work is the class conditional variant of Glow (Kingma& Dhariwal, 2018, Appendix D).

How is Eq (3),(10) come from?

**[Invert to Learn to Invert Patrick,NIPS19](https://arxiv.org/abs/1911.10914)**

**[A Disentangling Invertible Interpretation Network for Explaining Latent Representations,Arxiv2004](https://arxiv.org/pdf/2004.13166.pdf)**
 
**[IIR,Arxiv2005](https://arxiv.org/abs/2005.05650)**

**[Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation,Arxiv2003](https://arxiv.org/pdf/1308.3432.pdf)**

**[Integer Discrete Flows and Lossless Compression,NeuIPS19](https://papers.nips.cc/paper/9383-integer-discrete-flows-and-lossless-compression)**

intuition the latent variable is discrete?

**[Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement,ICLR19,oral](https://arxiv.org/pdf/1903.06059.pdf)**

## Summary

**discrete or continuous**

In Glow, they mentioned if x is discrete data, the log-likelihood objective is simply as:

$$
L(D) = \frac{1}{N} \sum_{i=1}^{N} -logp_{\theta}(x^{(i)})
$$

If x is  a continuous data(narual images are in this case, therefore we need dequantization. check PixelRNN):

$$
L(D) = \frac{1}{N} \sum_{i=1}^{N} - logp_{\theta}(\tilde{x}^{(i)}) + c
$$In Glow, they mentioned if x is discrete data, the log-likelihood objective is simply as:

$$
L(D) = \frac{1}{N} \sum_{i=1}^{N} -logp_{\theta}(x^{(i)})
$$

If x is  a continuous data(narual images are in this case, therefore we need dequantization. check PixelRNN):

$$
L(D) = \frac{1}{N} \sum_{i=1}^{N} - logp_{\theta}(\tilde{x}^{(i)}) + c
$$

**Prior choice summary**

The essential problem is how to obtain p(z) if you know the value of z, a necesary prior is you need the distribution type of z!



In realNVP, they set p(x) to be an isotropic unit norm Gaussian.

In NICE, the prior distribution p(x) can be gaussian distribution:

$$
log(p(x)) = -\frac{1}{2} (x^{2} + log(2\pi))
$$

or logistic distribution:

$$
log(p(x)) = -log(1+exp(x)) - log(1+exp(-x))
 $$

 They tend to use the logistic distribution as it tends to provide a better behaved gradient.


How about Glow?

In PixelRNN,they use discrete categorical distribution formulated by softmax.

In Image-Transformer:

> experiment with two settings of the distribution: a categorical distribution across
each channel (van den Oord et al., 2016a) and a mixture of discretized logistics over three channels (Salimans et al.). **[the categorical distribution is a special case of the multinomial distribution, in that it gives the probabilities of potential outcomes of a single drawing rather than multiple drawings.]**


As suggested by NICE, the prior distribution is factorial. we can simply multiply the prior distribution of every dimension(or pixel), then take logarithm, which is equavalent of sum up of log of each dimension's prior distribution. 

In summary: the typical choice is gaussian, logistic distribution


**Pixel processing summary**

In realNVP, they mentioned this:
> In order to reduce the impact of boundary effects(the boundary effects here can be seen in Figure 6 of PixelRNN.), we instead model the density of logit(α+(1−α)
x/256 ), where α is picked here as .05.

In PixelRNN, they also discussed quite a lot about the continous or discrete variable we should see the pixel as. They see them as a discrete variable and model them with a softmax layer(see Figure 6).
But previous models are trained with continuous variables, here they show how they compare with previous works:
> All our models are trained and evaluated on the loglikelihood loss function coming from a discrete distribution. **Although natural image data is usually modeled with
continuous distributions using density functions**, we can compare our results with previous art in the following way. In the literature it is currently best practice to add realvalued noise to the pixel values to dequantize the data when using density functions (Uria et al., 2013). When uniform noise is added (with values in the interval [0, 1]), then the log-likelihoods of continuous and discrete models are directly comparable (Theis et al., 2015). In our case, we can use the values from the discrete distribution as a piecewiseuniform continuous function  that has a constant value for every interval [i, i + 1], i = 1, 2, . . . 256. This corresponding distribution will have the same log-likelihood (on data with added noise) as the original discrete distribution (on discrete data).


In MAF:

> For both MNIST and CIFAR-10, we use the same preprocessing as by Dinh et al. [6]. We dequantize
pixel values by adding uniform noise, and then rescale them to [0, 1]. We transform the rescaled pixel
values into logit space by x 7→ logit(λ + (1 − 2λ)x), where λ= 10−6
for MNIST and λ= 0.05 for CIFAR-10, and perform density estimation in that space. In the case of CIFAR-10, we also augment the train set with horizontal flips of all train examples (as also done by Dinh et al. [6]).

In Image-Transformer, they didn't consider the difference between continuous and discrete data.

In RNADE:

> Pixels in this dataset can take a finite number of brightness values ranging from 0 to 255. Modeling
discretized data using a real-valued distribution can lead to arbitrarily high density values, by locating narrow high density spike on each of the possible discrete values. In order to avoid this ‘cheating’ solution, we added noise uniformly distributed between 0 and 1 to the value of each pixel. We then divided by 256, making each pixel take a value in the range [0, 1].

**Evaluation metrics summary**

for bits-per-dimension check page12 of MAF.

if your goal is density estimation(Glow, MAF,MADE,RealNVP,Sylvester):

- minimize NLL(in nats)
  
If your goal is vairiational inference, you can evaluate on ELBO, and NLL. To obtain NLL, you need importance sampling.


> Sylvester flow: In order to obtain estimates for the negative log likelihood we used importance sampling (as proposed in (Rezende et al., 2014)).Unless otherwise stated, 5000 importance samples were used.


## Coding

toy example,TODO




