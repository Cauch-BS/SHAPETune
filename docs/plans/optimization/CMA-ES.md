# The CMA-ES Package (by *CyberAgentAI*)
*Link to Paper*: [`cmaes` : A Simple yet Practical Python Library for CMA-ES](https://arxiv.org/pdf/2402.01373)

## Introduction:
CMA-ES is a tool, primarily a method of solving the [black-box optimization](./Black-Box.md) problem for *continuous* inputs. It optimizes the *objective* function by sampling candidate solutions from a multivariate Gaussian distribution, and is optimized for *parallelism*. It is particularly superior in cases where the function is ill-conditioned, non-separable, or rugged. 

Following the structure of the paper above, this summary is divided into 3 sections. 
- *Section 1*: Technical Specifics of the CMA-ES Algorithm
- *Section 2*: The Design Philosophy behind `cmaes`
- *Section 3*: Recent advances in CMA-ES integrated into `cmaes`

### Related Works:
By Nomura and Shibata's own admission, the [`pycma`](https://github.com/CMA-ES/pycma) implementation by Hansen et al. contains a more comprehensive suite of CMA-ES features. However, the `cmaes` library focuses on more basic and essential features, with a focus on simplicity and ease of understanding. 

Meanwhile, [`evojax`](https://github.com/google/evojax/tree/main/evojax/algo) and [`evosax`](https://github.com/RobertTLange/evosax) are [`JAX`](https://github.com/google/jax) based libraries which are ideal for leveraging GPUs and TPUs in scalable optimization. The relevant implementations can be found for [`evojax`'s `cma_jax`](https://github.com/google/evojax/blob/main/evojax/algo/cma_jax.py) and [`evosax`'s `cma_es`](https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/cma_es.py). 

## Section 1: Technical Specifications of CMA-ES
We consider the minimization of a function $f: \mathbb{R}^{d} \mapsto \mathbb{R}$. CMA-ES optimizes $f$ by sampling solutions from the Gaussian distribution $\mathcal{N}(\mathbf{m}, \sigma^2 \mathbf{C})$, where $\mathbf{m}$ is the mean vector in $\mathbb{R}^{d}$, $\sigma$ is the step size and $\mathbf{C} \in \mathbb{R}^{d \times d}$ is the covariance matrix. The way CMA-ES works is by updating $\mathbf{m}$, $\mathbf{\sigma}$, and $\mathbf{C}$ in response to evaluations of $f$. 

### Step 1: Sampling and Ranking
For the $(g+1)^{\text{st}}$ generation (with $g$ starting from $0$), for the population size $\lambda$, $\lambda$ candidate solutions $x_i$ ($i = 1, 2, \cdots, \lambda$) are independently sampled from $\mathcal{N}(\mathbf{m}^{(g)}, \left(\sigma^{(g)}\right)^2 \mathbf{C}^{(g)})$. These solutions are then sorted in ascending order by the evaluation result of $f$. 

$$ \begin{align}
\mathbf{z}_i &\sim \mathcal{N}({0, \mathbf{I}}) \\
\mathbf{y}_i &= \sqrt{\mathbf{C}^{(g)}}\mathbf{z}_i \\
\mathbf{x}_i &= \mathbf{m}^{(g)} + \sigma^{(g)}\mathbf{y}_i \\
{1:\lambda}, {2:\lambda}, \cdots, {\lambda:\lambda} &= \text{ArgSort}(f(\mathbf{x}_i))
\end{align}$$

### Step 2: Update Evolution Path
Using the parent number $\mu \le \lambda$ and the weights $\mathbf{w} = (w_i)_{1 \le w_i \le \mu}$ which satisfy the conditions
$$ i < j \implies w_i \le w_j$$
$$\sum_{i=1}^{\mu} w_i = 1$$

We calculate the weighted average for the $\mu$ best solutions: $\mathrm{d}\mathbf{y} = \sum_{i=1}^{\mu} w_i\mathbf{y}_{i:\lambda}$. The evolution path $\rho$ is then updated for $\sigma$ and $\mathbf{C}$:
$$\begin{align}
\rho_{\sigma}^{(g+1)} &= (1 - c_{\sigma}) \rho_{\sigma}^{(g)} + \dfrac{\sqrt{c_\sigma(2- c_\sigma)}}{\|\mathbf{w}\|}\sqrt{\mathbf{C}^{(g)}}^{-1} \mathrm{d}\mathbf{y} \\
\rho_{\mathbf{C}}^{(g+1)} &= (1 - c_{\mathbf{C}}) \rho_{\mathbf{C}}^{(g)} + \dfrac{\sqrt{c_\mathbf{C}(2- c_\mathbf{C})}}{\|\mathbf{w}\|}  H(\sigma^{(g+1)}) \mathrm{d} \mathbf{y}
\end{align}$$
where $c_{\mathbf{C}}, c_{\sigma}$ are fixed constants known as *cumulation* factors for $\mathbf{C}$ and $\mathbf{\sigma}$ and $H$ is a modified heavyside step function defined as

$$
H(\sigma^{g+1}) = 
\begin{cases} 
1 & \text{if } \dfrac{\|\rho_{\sigma}^{g+1}\|}{\sqrt{1 - (1- c_\sigma)^{2(g+1)}}\mathbb{E}[\|\mathcal{N}(0, \mathbf{I})\|]} < 1.4 + \dfrac{2}{d+1}\\
0 & \text{otherwise }
\end{cases}
$$

where $d$ is the dimension of the domain of $f$, and $\mathbb{E}[\|\mathcal{N}(0, \mathbf{I})\|]$ — i.e. the expected norm of samples from $\mathcal{N}(0, \mathbf{I})$ — is approximated as $\sqrt{d}\left( 1 - \dfrac{1}{4d} + \dfrac{1}{21d^2} \right)$. 

### Step 3: Update Parameters

The distribution parameters are updated by the following recursive relation:
$$
\begin{align}
\mathbf{m}^{(g+1)} &= \mathbf{m}^{(g)} + c_\mathbf{m} \sigma^{(g)} \mathrm{d}\mathbf{y} \\
\sigma^{(g+1)} &= \sigma^{(g)} \exp \left( \frac{c_\sigma}{d_\sigma} \left(\frac{\|p_\sigma^{(g+1)}\|}{\mathbb{E}[\|\mathcal{N}(0,I)\|]} - 1\right)\right) \\
\mathbf{C}^{(g+1)} &= \left(1 + (1 - H(\sigma^{g+1}))c_1 c_{\mathbf{C}}(2 - c_{\mathbf{C}})\right) \mathbf{C}^{(g)}\\
 &+ \underbrace{ c_1 \left[p_\mathbf{C}^{(g+1)} \left(p_{\mathbf{C}}^{(g+1)}\right)^\top - \mathbf{C}^{(g)}\right]}_{\text{rank-one update}}\\
&+ \underbrace{ c_\mu \sum_{i=1}^{\lambda} w_i^\circ \left[y_{i:\lambda}y_{i:\lambda}^\top - \mathbf{C}^{(g)}\right]}_{\text{rank-}\mu\text{ update}}
\end{align}
$$
the terms $H(\sigma^{g+1})$, $c_\sigma$, and $c_{\mathbf{C}}$ are deinfed above. The term $w_i^{o}$ is defined so that
$$
w_i^{o} = 
\begin{cases}
w_i & \text{if } w_i \ge 0 \\
\dfrac{w_id}{\left\|\sqrt{C^{(g)}}^{-1}(y_{i:\lambda})_{1 \le i \le \lambda}\right\|} & \text{else if } w_i < 0
\end{cases}
$$
where $d$ is the dimension of the domain of $f$. Meanwhile $c_m$ is called the *learning rate* for $\mathbf{m}$, $d_\sigma$ is called the damping factor for $\sigma$, and $\mathbf{c_1}$ and $\mathbf{c_\mu}$ are called the learning rates for rank-one updates and rank $\mu$ updates of $\mathbf{C}$ respectively. This method of adaption for $\sigma$ (i.e. $\text{Eq. }(8))$ is called the cumulative step-size adaptation. 

### Addendum 1: Relation to Gradient Descent

CMA-ES is equivalent to Gradient Descent on the *expected* **evolution-gradient**. For more information on mathematical details, see [the paper by Akimoto et al.](https://link.springer.com/chapter/10.1007/978-3-642-15844-5_16). 

### Addendum 2: Properties of CMA-ES
CMA-ES satisfies the following properties:
- order invariance : updates are equal under order-preserving transformations of the objective function
- affine invaraiance: updates are equal under affine transformations of the search space. 

This allows CMA-ES to be generalized to a wider range of problems. 

### Addendum 3: Further Reading
As our focus is on implementing CMA-ES, I have only summarized the algorithm briefly. For more details about the algorithm and the underlying mathematics behind it, read [the paper by Hansen](http://arxiv.org/abs/2402.01373) (author of the `pycma` package). 

## Section 2: The Design Philosophy and Practical Implementation of `cmaes`

`cmaes` was optimized for simplicity, with the only notable dependency being `numpy`. Meanwhile, for practicality `cmaes` incorporates the latest development in CMA-ES, implementing 3 state-of-the-art algorithms. 

1. *multimodal* CMA-ES (*multimodality*, in this case means minimizing a function with *multiple* local minima) through [**LRA-CMA**](./LRA-CMA.md).
2. *transfer-learning* CMA-ES (i.e. utilization of previous optimization results) by [**WS-CMA**](./WS-CMA.md)
3. Mixed-integer variable handling through [**CMAwM**](./CMAwM.md)

### Practical Implementation

Basic Usage: `cmaes` uses a basic `ask-and-tell` interface, where `ask` generates candidate solutions $x_\lambda$  and `tell` updates distribution parameters $\mathbf{m}, \sigma, \mathbf{C}$. 

```python
import numpy as np
from cmaes import CMA

optim = CMA(mean = np.zeros(2), sigma = 2)
for generation in range(num_generations):
    solutions = []
    for _ in range(optim.population_size):
        cand = optim.ask() # generates candidates
        value = objective(x)
        solutions.append(x, value)
    optim.tell(solutions) #updates parameters
```

## Section 3: Incorporating Recent Advances in CMA-ES

The code implements CMA-ES methodologies such as [**LRA-CMA**](./LRA-CMA.md),[**WS-CMA**](./WS-CMA.md), and [**CMAwM**](./CMAwM.md).