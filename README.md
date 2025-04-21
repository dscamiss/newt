# `newt` :lizard:

![License](https://img.shields.io/badge/license-MIT-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Build](https://github.com/dscamiss/newt/actions/workflows/python-package.yml/badge.svg)
[![codecov](https://codecov.io/gh/dscamiss/newt/graph/badge.svg?token=Z3CGGZJ70B)](https://codecov.io/gh/dscamiss/newt)

# Introduction

This package provides a PyTorch implementation of a "Newton-like" learning rate scheduler.

The general approach [1] is to attempt to minimize the loss function $L : \Theta \to \mathbb{R}$ by iterating

$$
\begin{align*}
    \theta_{t+1} = \theta_t - \alpha_t u_t \\
    \alpha_{t+1} = \alpha_t - \frac{g'_t(\alpha_t)}{g''_t(\alpha_t)},
\end{align*}
$$

where

* $\alpha_t$ is the learning rate at iteration $t$,
* $u_t$ is the $\theta$ update vector at iteration $t$, and
* $g_t(\alpha) = L(\theta_t - \alpha u_t)$.

In other words, we simultaneously run a gradient descent update on $\theta$ (using an arbitrary
optimizer to produce the update vectors) and a Newton update on $\alpha$.  

The implementation details primarily concern the Newton update, since directly computing $g''_t(\alpha_t)$ 
requires an expensive Hessian-vector product.  To work around this, we must use an approximation.
Our choice of approximation, compared with the one in [1], is described [here](https://dscamiss.github.io/blog/posts/newton-like-method/).

# Installation

In an existing Python 3.9+ environment:

```python
git clone https://github.com/dscamiss/newt
pip install ./newt
```

# Usage

TODO

# References

1. G. Retsinas, G. Sfikas, P. Filntisis and P. Maragos, "Newton-Based Trainable Learning Rate," ICASSP 2023.
2. G. Retsinas, G. Sfikas, P. Filntisis and P. Maragos, "Trainable Learning Rate",
2022, retracted.
