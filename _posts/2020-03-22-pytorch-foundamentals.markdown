---
layout: draft
title: "All kinds of pytorch backward"
date: 2020-03-22 14:49:0 +0000
comments: False
share: False
categories: ml
---

## Background in pytorch

**Tensor** and **torch.autograd.Function** are interconnected and build up an acyclic graph, that encodes a complete history of computation. The computation graph is interconnected by **grad_fn** attribute in pytorch.


[https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)


## torch.nn.functional

Typical functional can be found at [https://pytorch.org/docs/stable/nn.functional.html](https://pytorch.org/docs/stable/nn.functional.html). 

The functional versions are stateless, and called directly, eg for softmax, which has no internal state.
```
x = torch.nn.functional.soft_max(x)
```
There are functional versions of various stateful network modules. In this case, you have to pass in the state yourself. Conceptually, for Linear, it’d be something (conceptually) like:
```
x = torch.nn.functional.linear(x, weights_tensor)
```

The difference between torch.nn and torch.nn.functional is a matter of convenience and taste. torch.nn is more convenient for methods which have learnable parameters.

All modules you used in pytorch, if we trace them back inside the python level, they are all actually invoking different kinds of torch.nn.functional methods. you can check a demo here:

[https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html](https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html)

In this example, MyReLU, [mm](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.mm), [pow](https://pytorch.org/docs/stable/torch.html#torch.pow), sum.

Typical invoke order:

- torch.nn->torch.nn.functional->torch.tensor.ops->torch._C.nn.ops
- torch.nn->torch.tensor.ops->torch._C.nn.ops



Typical implementation(invoke) pattern in torch.nn.functional:

- Some directly invoke torch.tensor.ops, such as [torch.nn.functional.sigmoid](https://github.com/pytorch/pytorch/blob/7468ef04c26787fd647c06cf98703f9f281f1715/torch/nn/functional.py#L1562)
- Some will invoke C++ code, such as [torch.nn.functional.conv1d](https://github.com/pytorch/pytorch/blob/7468ef04c26787fd647c06cf98703f9f281f1715/torch/nn/functional.py#L20)
- Some will invoke tensor.ops, such as [torch.nn.functional.relu](https://github.com/pytorch/pytorch/blob/7468ef04c26787fd647c06cf98703f9f281f1715/torch/nn/functional.py#L1050)
- Some will invoke torch._C._nn.ops, such as [torch.nn.functional.glu](https://github.com/pytorch/pytorch/blob/7468ef04c26787fd647c06cf98703f9f281f1715/torch/nn/functional.py#L1074)
- Some will directly be implemented in torch.nn.functional, such as [torch.nn.functional.gumbel_softmax](https://github.com/pytorch/pytorch/blob/7468ef04c26787fd647c06cf98703f9f281f1715/torch/nn/functional.py#L1448)

## torch.autograd.Function

https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function


https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html


  

## Atomic Operation

typical grad_fn is: AddBackward,PowBackward,PowBackward,AddmmBackward,DivBackward, etc.

For example:
```
z =x+1
print(z.grad_fn)
#AddBackward
```

```
y = torch.nn.Linear(2, 2)(x)
print(y,y.grad_fn)
#AddmmBackward
```
Here grad_fn is AddmmBackward, because Addmm is the last atomic operation in ``Linear''. A similar case is:
```
y = torch.nn.functional.cosine_similarity(x,x)
print(y.grad_fn)
#DivBackward0
```

## Why sometimes backward is unneccesary in function implementation?

When we write a atomic operation, we need a backward function to realize autograd. Sometimes, a function doesn't need backward implementation, because:

- This function is a high-level(non-atomic) function, because this function can be realized by other differentiable atomic operations(functions). such as typical Conv1D,Linear operation.
- This function doesn't have the input for differential.
- This function doesn't have the output for differential.
  

[torch.nn.CosineSimilarity](https://pytorch.org/docs/stable/nn.html?highlight=cosinesimilarity#torch.nn.CosineSimilarity) in pytorch id differentiable, see math [prove](https://math.stackexchange.com/questions/1923613/partial-derivative-of-cosine-similarity). Notice, pytorch only is responsible for gradient computing, this is different from optimization, they can not gurantee your model can be optimized as you wish.

## Inplace=True

inplace=True means that it will modify the input directly, without allocating any additional output(extra node/vertex in  computation graph). It can sometimes slightly decrease the memory usage, but may not always be a valid operation (because the original input is destroyed). However, if you don’t see an error, it means that your use case is valid.


## different backward function in pytorch

A useful tool to create visualizations of PyTorch execution graphs and traces: [pytorchviz](https://github.com/szagoruyko/pytorchviz)



**torch.nn.functional.soft_max**

this is a mapping function from $$B \times C$$ to $$B \times C$$, the derivation is splittable along all dimensions, therefore we can simply consider the scalar case,
consider $$y_{i}=\frac{exp(x_{i})}{\sum_{j \in C} exp(x_{j})}$$

$$\frac{\partial{y_{i}}}{\partial{x_{i}}} = y_{i}^{2} - y_{i}$$

You can also illustrate it by $$x_{i}$$ by replacing $$y_{i}$$ to $$x_{i}$$, the vector form is:

$$\frac{\partial{Y}}{\partial{X}} = Y^{2} - Y=(sigmoid(X))^{2} - sigmoid(X)$$

**torch.log**

$$y_{i}=log_{e} (x_{i})$$

the gradient is 

$$\frac{\partial{y_{i}}}{\partial{x_{i}}} = \frac{1}{x_{i}}$$



**nn.LogSoftmax**

$$y_{i}=log(softmax(z_{i})) = log(\frac{exp(x_{i})}{\sum_{j} exp(x_{j})})$$

This is a combination of nn.LogSoftmax and torch.nn.functional.soft_max, use chain rule can solve it.

**torch.NLLLoss**

torch.NLLLoss + nn.LogSoftmax = CrossEntropy Loss


**torch.nn.functional.cross_entropy**

$$L= - \sum_{j}^{C} y_{j} log(softmax(x_{j})) = - \sum_{j}^{C} y_{j} log(\frac{exp(x_{y})}{\sum_{m} exp(x_{m})})$$

let $$z_{j}= \frac{exp(x_{y})}{\sum_{m} exp(x_{m})}$$

$$\frac{\partial{L}}{\partial{x_{j}}} = z_{j}-1$$  if $$y_{j}=1$$,
else $$\frac{\partial{L}}{\partial{x_{j}}} = z_{j}$$

check [here](https://arxiv.org/pdf/2003.05176.pdf)

**torch.cat**

concatenation can be operated between diverse dimensions. As here we only reorganize the tensor and don't change the value, all gradients are just the reverse-organsed(with value unchanged).


**torch.mean**


**Convolution is equivalent with Unfold + Matrix Multiplication + Fold (or view to output shape)**

**[https://colab.research.google.com/drive/10aVzydfnBYWJKXe5SCTxGx0t529kufRd](https://colab.research.google.com/drive/10aVzydfnBYWJKXe5SCTxGx0t529kufRd)**

**Conv1d and nn.Linear**

[https://stackoverflow.com/questions/55576314/conv1d-with-kernel-size-1-interpretation/56685503#56685503](https://stackoverflow.com/questions/55576314/conv1d-with-kernel-size-1-interpretation/56685503#56685503)

**1x1 Conv2d and Linear**

[Can Fully Connected Layers be Replaced by Convolutional Layers?](https://sebastianraschka.com/faq/docs/fc-to-conv.html)

**softmax implementation numerical stability**

Imagine exp(1e20)+exp(1e10)
```
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

scores = [3.0, 1.0, 0.2]
print(softmax(scores))

```

## Reference

- [colab](https://colab.research.google.com/drive/1US3uQNTWUse1-D_4oK5TlKBRfAbrZmxD)
- [https://zhuanlan.zhihu.com/p/83517817](https://zhuanlan.zhihu.com/p/83517817)
- [https://zhuanlan.zhihu.com/p/56924766](https://zhuanlan.zhihu.com/p/56924766)

