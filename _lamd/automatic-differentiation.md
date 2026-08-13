---
layout: lecture
title: More Generalisation and Automatic Differentiation
week: 2
session: 1
author:
- given: Ferenc
  family: Huszár
  institution: University of Cambridge
  url: https://www.inference.vc/about/
abstract: >
  This lecture will cover the foundations of automatic differentiation as well as the different frameworks that exist for building models.
talkscam:
venue: LT1, William Gates Building
youtube: -9O5obQZUn0
oldyoutube: 
- code: m3KZLPed7aM
  year: 2021
hackmdslides: fhuszar/H1WZ70kl_#
hackmdnotes: fhuszar/SyHTInWeu
hackmdworksheet: fhuszar/S1UdOvZe_
time: "14:00"
start: "14:00"
end: "15:00"
date: 2022-01-27
---

\notes{* [Slides are here](https://hackmd.io/@fhuszar/H1WZ70kl_#)
* [Approximating with ReLU notes are here](https://hackmd.io/@fhuszar/S1UdOvZe_)
* [Automatic Differentiaton notes are here](https://hackmd.io/@fhuszar/SyHTInWeu)
* [Google Colab Is Available Here](https://colab.research.google.com/drive/1qioPLq-dxOwudPKXU3MxpHr2s4Su3dxI?usp=sharing)


* Nice review paper from <https://jmlr.org/papers/v18/17-468.html> @Baydin-autodiff18 (See Figure 2)}

\subsection{Reddit Group}

[https://www.reddit.com/r/CST_DeepNN/](https://www.reddit.com/r/CST_DeepNN/)


\subsection{Gradient-based optimization}

$$
\mathcal{L}(\theta) = \sum_{n=1}^N \ell(y_n, f(x_n, \theta))
$$

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta)
$$

\subsubsection{Basic Multilayer Perceptron}

$$
\small
f_L(x) = \phi\left(b_L + W_L \phi\left(b_{L-1} + W_{L-1} \phi\left( \cdots \phi\left(b_1 + W_1 x\right) \cdots \right)\right)\right)
$$

\subsection{General deep function composition}

$$
f_L(f_{L-1}(\cdots f_1(\mathbb{w})))
$$

How do I calculate the derivative of $f_L(\mathbb{w})$ with respect to $\mathbb{w}$?

\subsection{Chain rule}

$$
\frac{\partial \mathbf{f}_L}{\partial \mathbb{w}} = \frac{\partial \mathbf{f}_L}{\partial \mathbf{f}_{L-1}} \frac{\partial \mathbf{f}_{L-1}}{\partial \mathbf{f}_{L-2}} \frac{\partial \mathbf{f}_{L-2}}{\partial \mathbf{f}_{L-3}} \cdots \frac{\partial \mathbf{f}_3}{\partial \mathbf{f}_{2}} \frac{\partial \mathbf{f}_2}{\partial \mathbf{f}_{1}} \frac{\partial \mathbf{f}_1}{\partial \mathbf{w}} 
$$

\subsection{How to evaluate this?}

$$
\frac{\partial \mathbf{f}_L}{\partial \mathbb{w}} = \frac{\partial \mathbf{f}_L}{\partial \mathbf{f}_{L-1}} \frac{\partial \mathbf{f}_{L-1}}{\partial \mathbf{f}_{L-2}} \frac{\partial \mathbf{f}_{L-2}}{\partial \mathbf{f}_{L-3}} \cdots \frac{\partial \mathbf{f}_3}{\partial \mathbf{f}_{2}} \frac{\partial \mathbf{f}_2}{\partial \mathbf{f}_{1}} \frac{\partial \mathbf{f}_1}{\partial \mathbf{w}} 
$$

$$
\small
\frac{\partial \mathbf{f}_L}{\partial \mathbb{w}} = \frac{\partial \mathbf{f}_L}{\partial \mathbf{f}_{L-1}} \left( \frac{\partial \mathbf{f}_{L-1}}{\partial \mathbf{f}_{L-2}} \left( \frac{\partial \mathbf{f}_{L-2}}{\partial \mathbf{f}_{L-3}} \cdots \left( \frac{\partial \mathbf{f}_3}{\partial \mathbf{f}_{2}} \left( \frac{\partial \mathbf{f}_2}{\partial \mathbf{f}_{1}} \frac{\partial \mathbf{f}_1}{\partial \mathbf{w}} \right) \right) \cdots \right) \right)
$$

\subsection{Or like this?}

$$
\frac{\partial \mathbf{f}_L}{\partial \mathbb{w}} = \frac{\partial \mathbf{f}_L}{\partial \mathbf{f}_{L-1}} \frac{\partial \mathbf{f}_{L-1}}{\partial \mathbf{f}_{L-2}} \frac{\partial \mathbf{f}_{L-2}}{\partial \mathbf{f}_{L-3}} \cdots \frac{\partial \mathbf{f}_3}{\partial \mathbf{f}_{2}} \frac{\partial \mathbf{f}_2}{\partial \mathbf{f}_{1}} \frac{\partial \mathbf{f}_1}{\partial \mathbf{w}} 
$$

$$
\small
\frac{\partial \mathbf{f}_L}{\partial \mathbb{w}} = \left( \left( \cdots \left( \left( \frac{\partial \mathbf{f}_L}{\partial \mathbf{f}_{L-1}} \frac{\partial \mathbf{f}_{L-1}}{\partial \mathbf{f}_{L-2}}  \right) \frac{\partial \mathbf{f}_{L-2}}{\partial \mathbf{f}_{L-3}} \right) \cdots \frac{\partial \mathbf{f}_3}{\partial \mathbf{f}_{2}} \right) \frac{\partial \mathbf{f}_2}{\partial \mathbf{f}_{1}} \right) \frac{\partial \mathbf{f}_1}{\partial \mathbf{w}} 
$$

\subsection{Or in a funky way?}

$$
\frac{\partial \mathbf{f}_L}{\partial \mathbb{w}} = \frac{\partial \mathbf{f}_L}{\partial \mathbf{f}_{L-1}} \frac{\partial \mathbf{f}_{L-1}}{\partial \mathbf{f}_{L-2}} \frac{\partial \mathbf{f}_{L-2}}{\partial \mathbf{f}_{L-3}} \cdots \frac{\partial \mathbf{f}_3}{\partial \mathbf{f}_{2}} \frac{\partial \mathbf{f}_2}{\partial \mathbf{f}_{1}} \frac{\partial \mathbf{f}_1}{\partial \mathbf{w}} 
$$

$$
\small
\frac{\partial \mathbf{f}_L}{\partial \mathbb{w}} = \frac{\partial \mathbf{f}_L}{\partial \mathbf{f}_{L-1}} \left( \left( \left( \frac{\partial \mathbf{f}_{L-1}}{\partial \mathbf{f}_{L-2}}  \right) \frac{\partial \mathbf{f}_{L-2}}{\partial \mathbf{f}_{L-3}} \right) \left( \left( \cdots \frac{\partial \mathbf{f}_3}{\partial \mathbf{f}_{2}} \right) \frac{\partial \mathbf{f}_2}{\partial \mathbf{f}_{1}} \right) \right)\frac{\partial \mathbf{f}_1}{\partial \mathbf{w}} 
$$

\subsection{Automatic differentiation}

**Forward-mode**

$$
\small
\frac{\partial \mathbf{f}_L}{\partial \mathbb{w}} = \frac{\partial \mathbf{f}_L}{\partial \mathbf{f}_{L-1}} \left( \frac{\partial \mathbf{f}_{L-1}}{\partial \mathbf{f}_{L-2}} \left( \frac{\partial \mathbf{f}_{L-2}}{\partial \mathbf{f}_{L-3}} \cdots \left( \frac{\partial \mathbf{f}_3}{\partial \mathbf{f}_{2}} \left( \frac{\partial \mathbf{f}_2}{\partial \mathbf{f}_{1}} \frac{\partial \mathbf{f}_1}{\partial \mathbf{w}} \right) \right) \cdots \right) \right)
$$

Cost: 
$$
\small
d_0d_1d_2 + d_0d_2d_3 + \ldots + d_0d_{L-1}d_L = d_0 \sum_{l=2}^{L}d_ld_{l-1}
$$

\subsection{Automatic differentiation}

**Reverse-mode**

$$
\small
\frac{\partial \mathbf{f}_L}{\partial \mathbb{w}} = \left( \left( \cdots \left( \left( \frac{\partial \mathbf{f}_L}{\partial \mathbf{f}_{L-1}} \frac{\partial \mathbf{f}_{L-1}}{\partial \mathbf{f}_{L-2}}  \right) \frac{\partial \mathbf{f}_{L-2}}{\partial \mathbf{f}_{L-3}} \right) \cdots \frac{\partial \mathbf{f}_3}{\partial \mathbf{f}_{2}} \right) \frac{\partial \mathbf{f}_2}{\partial \mathbf{f}_{1}} \right) \frac{\partial \mathbf{f}_1}{\partial \mathbf{w}} 
$$

Cost:
$$
\small
d_Ld_{L-1}d_{L-2} + d_{L}d_{L-2}d_{L-3} + \ldots + d_Ld_{1}d_0 = d_L \sum_{l=0}^{L-2}d_ld_{l+1}
$$

\subsection{Automatic differentiation}

* in deep learning we're most interested in scalar objectives
* $d_L=1$, consequently, backward mode is always optimal
* in the context of neural networks: **backpropagation**
* backprop has higher memory cost than forwardprop

\subsection{Example: calculating a Hessian}

$$
H(\mathbb{w}) = \frac{\partial^2}{\partial\mathbf{w}\partial\mathbf{w}^\top} L(\mathbf{w})
= \frac{\partial}{\partial\mathbf{w}} \mathbf{g}(\mathbf{w})
$$

\subsection{Example: Hessian-vector product}

$$
\mathbf{v}^\top H(\mathbf{w}) = \frac{\partial}{\partial\mathbf{w}} \left( \mathbf{v}^\top \mathbf{g}(\mathbf{w}) \right)
$$

}
