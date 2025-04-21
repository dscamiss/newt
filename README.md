# `newt` :lizard:

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

In other words, we simultaneously run a gradient descent update on $\theta$
and a Newton update on $\alpha$.  The implementation details primarily concern the Newton update,
since it involves the second-order derivative $g''_t(\alpha_t)$.  An exact computation of this quantity requires expensive Hessian-vector products, 
so an approximation is necessary.  

There is some freedom in the nature of the approximation.  Our approach, 
compared with the approach of [1], is described [here](https://dscamiss.github.io/dscamiss/newton-like-method/).

# Installation

In an existing Python 3.9+ environment:

```python
git clone https://github.com/dscamiss/newt
pip install ./newt
```

# Usage

TODO

# References

1. G. Retsinas, G. Sfikas, P. Filntisis and P. Maragos, "Newton-Based Trainable Learning Rate," ICASSP 2023
2. G. Retsinas, G. Sfikas, P. Filntisis and P. Maragos, "Trainable Learning Rate",
2022, retracted.
