---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: pyDCI
    language: python
    name: pydci
---

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## Preamble
<!-- #endregion -->

```python
from pydci.log import enable_log
import numpy as np
import importlib
import matplotlib.pyplot as plt
from scipy.stats.distributions import uniform, norm
from pydci import ConsistentBayes as CB

__author__ = "Carlos del-Castillo-Negrete"
__copyright__ = "Carlos del-Castillo-Negrete"
__license__ = "mit"

seed = 123456

_ = enable_log()
```

<a id='Title'></a>
# <center> Data-Consistent Inversion Class
_____

<center>
    Notebook by:
    <br>
    Carlos del-Castillo-Negrete (<font color='blue'>cdelcastillo21@gmail.com</font>)
</center>


The DCIProblem class is the main class all other methods that implement DCI algorithms inerhit off of.


<!-- #region slideshow={"slide_type": "slide"} -->
<a id='Title'></a>
## Data-Consistent Update Formula: 

The DCIProblem class solution revolves around computing the update.

$\Large \pi^{up}_{\Lambda}(\lambda) = \pi^{in}_{\Lambda}(\lambda)\frac{\pi^{obs}_{\mathcal{D}}(Q(\lambda))}{\pi^{pred}_{\mathcal{D}}(Q(\lambda))}$

The inputs for the class

1. $\lambda$  - `lam` - Initial samples. 2D array of dimensions (num_samples x param_dimensions)
2. $Q(\lambda)$ - `q_lam` - Samples pushed through forward map. 2D array of dimensions (num_samples x number_output_states)
3. $\pi^{obs}_{\mathcal{D}}$ - `pi_obs` - Observed distribution over data.

Optionally as well one can specify:

4. $\pi^{in}_{\Lambda}$ - `pi_in` - Initial distribution over samples. If not specified, then computed via a gaussain kernel density estiamte on `lam`, as implemented by scipy.
5. $\pi^{pr}_{\mathcal{D}}$ - `pi_pr` - Predicted distribution, i.e. the distribution that catagorizes the push-forwardo the samples through the forward model. If not specified, then `pi_pr` is computed via a gaussian kernel density estimate on `q_lam`.
6. `weights` - Weights to assignto each `lam` sample, to incorporate prior beliefs. This is used more by inherited classes, such as the `SequentialMUDProblem` class that uses information from prior iterations to inform the current one.

Note: For all distribution arguments `pi_*`, they can be specified as a scipy.stats distribution (see https://docs.scipy.org/doc/scipy/reference/stats.html), or as a scipy gaussian kernel density (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html).
<!-- #endregion -->

## Example 1 - Low Dimensional Example
---


We start with a simple QoI map over a one-dimensional parameter space:
   
   $$ \Large \Lambda = [-1, 1] \in \mathbb{R} $$  
   
   $$ \Large Q(\lambda) = \lambda^5 \rightarrow \mathcal{D} = [-1, 1] $$


We assume:

   $\pi_{in} \sim \mathcal{U}([-1, 1])$
  
   $\pi_{ob} \sim \mathcal{N}(0.25,0.1^2)$


### Data

We take $N$ samples from an initial uniform distribution over the input parameter space. These samples will be pushed forward through our QoI map to constrcut the predicted density $\pi_{pred}$:

```python
from scipy.stats.distributions import uniform, norm
import numpy as np

np.random.seed(123)

p, num_samples, mu, sigma, domain = 5, int(1e3), 0.25, 0.1, [-1, 1]
lam = uniform.rvs(size=(num_samples, 1), loc=domain[0],
                  scale=domain[1] - domain[0])
q_lam = (lam**p).reshape(num_samples, -1)
data = np.array([0.25])
lam.shape, q_lam.shape, data.shape
```

```python
from pydci import ConsistentBayes as CB
importlib.reload(CB)

d_prob = CB.DCIProblem((lam, q_lam), norm(data, scale=sigma))
d_prob.solve()
```

## State Dataframe

The central data structure managed by the ConsistentBayes class is the state pandas DataFrame. 
This DataFrame stores in each column the values necessary to compute the Data Consistent Bayes update formula:

$\Large \pi_{up}(\lambda) = \pi_{in}(\lambda)\frac{\pi_{ob}(Q(\lambda))}{\pi_{pred}(Q(\lambda))}$

Since we are approximating the probability densities with finite samples, we deal with not the densities explicitly, but rather their values evaluated at each parameter sample $\lambda$.
Each term in the update can be found in the state DataFrame in the following columns:
 
- lam_* - Parameter samples $\lambda$.
- q_lam_* Push-forward of parameter samples through forward model $Q(\lambda)$.
- pi_* - Distributions in data-consistent problem evaluate at each parameter sample, or push-forward of the sample.
- weight - Weight assigned to each parameter sample. By default set to 1.0 for each sample. Can be used to incorporate prior beliefs into the update.

```python
d_prob.state           # State stores all info on the DCI problem as solved
```

## Output Diagnostics

One strength of Data-Consistent Inversion, is the theory behind it gives us diagnostics to asses the quality of our solution.
The `result` attribute stores two such metrics:

1. $\mathbb{E}(r)$ -  The expected value of the ratio of the predicted and observed distributions, which should be close to 1 if the predictability assumption is satisfied. See (link) for more info
2. $\mathcal{D}_{KL}$ - the KL Divergence between the observed and the initial, which quantifies the information gain made by the data consistent inversion update.

```python
d_prob.result
```

## Solution - Plotting Densities

The DCI problem class provides a couple of native plotting functions to visualize the solutions computed.

```python
d_prob.density_plots()
```

```python
d_prob.state_plot()
```

## Explicitly Specifying `pi_in`

Note how in the case above, we knew that the initial samples came from a uniform distribution.
We could explicitly set it so that the pyDCI module doesn't have to do a kernel density estimate on the initial samples, making it run faster.

```python
from pydci import ConsistentBayes as CB
importlib.reload(CB)

pi_in = uniform(loc=domain[0], scale=domain[1] - domain[0])
d_prob = CB.DCIProblem((lam, q_lam), norm(data, scale=sigma), pi_in=pi_in)
d_prob.solve()
_ = d_prob.density_plots()
d_prob.result
```

## Explicitly Setting `pi_pr`

What if we knew analytically `pi_pr`? 
Or We've computed some other way via another kernel density estimate call.
As long as the passed in form takes in a .pdf() method similar to scip.stats distributions, then they can be passed to the DCIProblem constructor.

For example, in the case when the exponent `p = 1` in the monomial example, we simply have the identity map.
Then the predicted distribution, would just be the initial distribution in this case, and we could set it explictly.

```python
np.random.seed(123)

p, num_samples, data, sigma, domain = 1, int(1e3), [0.25], 0.1, [-1, 1]
lam = uniform.rvs(size=(num_samples, 1), loc=domain[0],
                  scale=domain[1] - domain[0])
q_lam = (lam**p).reshape(num_samples, -1)

pi_in = uniform(loc=domain[0], scale=domain[1] - domain[0])
d_prob = CB.DCIProblem((lam, q_lam), norm(data, scale=sigma), pi_in=pi_in, pi_pr=pi_in)
d_prob.solve()
_ = d_prob.density_plots()
```

```python

```
