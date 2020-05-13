---
layout: draft
title: "deep learning fundamentals"
date: 2020-03-22 14:49:0 +0000
comments: False
share: False
categories: cv
---

**im2col tricks**


[https://zhuanlan.zhihu.com/p/63974249](https://zhuanlan.zhihu.com/p/63974249)


Im2col size is [N(H-k+1)(W-k+1), kxkxc], convolution kernel size is [kxkxc, d].

The reason of naming 'im2col', for the size of im2col matrix, each column is N(H-k+1)(W-k+1) sized, it represents the whole image. There is also im2row, the size is [kxkxc, N(H-k+1)(W-k+1)].

When will im2row be used?

Why caffee is BCWH and torch is BWHC?

[Row- and column-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order)

[Why GEMM is at the heart of deep learning](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/)

[Low-memory GEMM-based convolution algorithms for deep neural networks](https://arxiv.org/pdf/1709.03395.pdf)

[efficient CPU version, as_strided](https://zhuanlan.zhihu.com/p/64933417)

[The Indirect Convolution Algorithm](https://arxiv.org/pdf/1907.02129.pdf)
[zhihu discussion](https://www.zhihu.com/question/336558535)


