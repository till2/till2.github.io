---
layout: post
title:  "Gaussian processes and bayesian optimization"
author: "Till Zemann"
date:   2023-02-18 01:09:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
tags: [machine learning, optimization]
thumbnail: "/images/gaussian_processes/GP_regression.png"
---


<div class="img-block" style="width: 700px;">
    <img src="/images/gaussian_processes/GP_regression.png"/>
</div>

<h3> Contents </h3>
* TOC
{:toc}

<!--
<em style="float:right">First draft: 2023-02-20</em><br>
-->

### Goal

In bayesian optimization, we want to optimize a target function $f$, which is costly to probe. A typical setting for this is hyperparameter optimization. Here, $f$ yields the empirical risk (average loss) of the model after training for a number of steps. 

During the optimization, we try out differerent vectors of hyperparameters. To balance the exploration-exploitation tradeoff (exploration: pick a region with high variance; exploitation: pick a region where we expect a high reward), we need an aquisition function, which gives us the next hyperparameter vector $\lambda$.

As input to our gaussian process, we are given a dataset $\mathcal{D}$ of probes, for example $\mathcal{D} = \{ (x_1, f(x_1)), (x_2, f(x_2)), (x_3, f(x_3)) \}$. The task is to predict the mean $\mu_\*$ for and variance $var_\*$ for the function value $f(x_\*)$ of a new input $x_\*$.

To do that, we assume that it follows the same distribution as the dataset: $f(x_\*) \sim \mathcal{N}(m(x_\*), k(x_\*, x_\*))$


### Gaussian Process

A gaussian process is a stochastic process that defines a stochastic value $y$ for every time step $t$. Therefore, it can be seen as a distribution over functions $f$ with $f(x) = y$. These functions are just vectors of function values $f(x)$ that can be sampled from the process.

To define a gaussian process $GP$, we need a mean function $m(x)$ and a kernel $k(x,x')$ as a covariance function:

$$
f(x) \sim GP( m(x), k(x,x') ).
$$

In the gaussian process, the mean and covariance are correlated with the other input data, as defined by the mean and covariance functions. This is more expressive than a simple multivariate gaussian, where the function values are jointly distributed and not correlated. 

We use a kernel function to model the covariance of two data points. By picking a specific kernel function, we put prior information into the model (e.g. how smooth the function is).


### Kernel function

In our gaussian process model, we assume that if two x values are close to each other, their corresponding y values with y=f(x) are also similar. Using this assumption, we can model the covariance (matrix) using a kernel function.
This function has to be positive-definite, which means among other things that it has a global maximum at $x=0  \implies \forall x. f(0) \geq f(x)$. This property must be satisfied because the covariance of two identical x has to be maximal.

One popular choice for the kernel is to use a __radial basis function (RBF)__, which looks like this:

$$
k(x,x') = \exp(-\frac{1}{2 \sigma^2} ||x-x'||^2) = 
\begin{cases}
 1 & \text{if } x=x' \\
\text{small } (\approx 0) & \text{if } x \text{ and } x' \text{ are far apart}\\
\end{cases}
$$

```py
def kernel_fn(x1,x2):
    """ RBF kernel function """
    sigma = 1
    return np.exp(-0.5 * sigma**2 * np.dot(x1-x2, x1-x2))
```

If we plot the RBF kernel function for a scalar variable x as the first argument and 0 as the second argument, we observe that it indeed has a global maximum at $x=0$ ($\rightarrow$ positive-definite ✅) and it drops off towards 0 as the x values get further apart.

<div class="img-block" style="width: 400px;">
    <img src="/images/gaussian_processes/radial_basis_function_kernel.png"/>
</div>


### Covariance matrix

Using our kernel function, we can model (predict) the __Covariance matrix__ $\Sigma$:

$$
\Sigma = \mathbb{E}[ (X-\mu_X) (Y-\mu_Y)^T ] \; \text{ with } \; \Sigma_{ij} = cov(X_i, Y_i)
$$

With the RBF kernel, the visualized $9 \times 9$ covariance matrix $\Sigma$ looks like this:

<div class="img-block" style="width: 300px;">
    <img src="/images/gaussian_processes/cov_matrix_9.png"/>
</div>

We can also use a finer matrix ($200 \times 200$) to get a smooth view at what the function does:

<div class="img-block" style="width: 300px;">
    <img src="/images/gaussian_processes/cov_matrix_200.png"/>
</div>

Just notice that for the two covariance matrices above, the kernel yields higher values (up to 1) for $x$ values that are close to each other. 


### Regression with the gaussian process

We use the $GP$ (which includes some assumptions in our kernel choice) as a prior and create a posterior distribution using our data. 
Using the posterior, we can predict the mean and variance of function values $\mathbf{y}_2$ for new data samples $X_2$. 

<div class="img-block" style="width: 700px;">
    <img src="/images/gaussian_processes/GP_regression.png"/>
</div>

$$
\begin{bmatrix} \mathbf{y}_1 \\ \mathbf{y}_2 \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix}, \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{bmatrix}\right)
$$


#### Solution

Mean of $\mathbf{y}_2$:

$$
\mu_{2|1} = (\Sigma_{11}^{-1} \Sigma_{12})^T \mathbf{y}_1
$$

Variance of $\mathbf{y}_2$:

$$
\Sigma_{2|1} = \Sigma_{22} - (\Sigma_{11}^{-1} \Sigma_{12})^{\top} \Sigma_{12} \\
$$


<!--
Correlation: 

$$cor(X,Y) = \frac{cov(X,Y)}{\sqrt{var(X) \cdot var(Y)}}$$
-->

<!--
Example with data X = 3x3 matrix
https://cdn.numerade.com/ask_images/e1ebad3023ae4fad824f457f9800c9b3.jpg
-->
<p class="vspace"></p>




### Bayesian Optimization Algorithm

```py
used_parameters = dict()

# maybe give it N hours to train and update the budget using wallclock time
budget = 10

while budget:
    hyperparam_vector = argmax(expected_improvement(GP))

    # train and evaluate model with the picked hyperparameters
    y = f(hyperparam_vector)

    used_parameters[hyperparam_vector] = y
    budget = budget - 1

# return hyperparam_vector with the minimal loss in f
```
- Draw a vector $f$ (which is basically a function if the vector has infinite length):

$$f(\text{hyperparam_vector}) \sim GP(0,k)$$

- Get the loss, which is stochastic (depends on SGD etc.):

$$\text{loss} \sim \mathcal{N}(0,\sigma^2)$$


### Exploration and Exploitation

__Exploration__: Prefers values that have a high posterior variance <br> (where we don't know much about the outcome yet).

__Exploitation__: Prefers values where the posterior has a high mean $\mu_x$.


### Aquisition function: Expected Improvement

To balance the exploration-exploitation tradeoff, we need an aquisition function that tells us which hyperparameter-vector $\lambda$ to pick next from the values in $X2$ when the data $X1, y1$ is given.

One possible aquisition function is __Expected Improvement (EI)__:

$$
A(\lambda \in X2 | X1,y1) = \mathbb{E}[ \max(0, y_{\max}) - y ] = 
\begin{cases}
(\mu - y_{\max} - a) \cdot CDF(Z) + \sigma \cdot PDF(Z) & \text{if}\ \sigma > 0 \\
0 & \text{if}\ \sigma = 0
\end{cases}
$$

with 

$$
Z =
\begin{cases}
\frac{\mu - y_{\max} - a}{\sigma} &\text{if}\ \sigma > 0 \\
0 & \text{if}\ \sigma = 0
\end{cases}
$$

and the hyperparameter $a$ for exploration (higher $a$ means more exploration). <br>
A common default value is $a = 0.01$ [[7]](http://krasserm.github.io/2018/03/21/bayesian-optimization/).



### Example for an Application

Bayesian optimization was used for hyperparameter tuning in the AlphaGo system, more details can be found in the [dedicated paper](https://arxiv.org/pdf/1812.06855.pdf). 


<!-- In-Text Citing -->
<!-- 
You can...
- use bullet points
1. use
2. ordered
3. lists


-- Math --
$\hat{s} = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \mu)^2$ 

-- Images --
<div class="img-block" style="width: 800px;">
    <img src="/images/lofi_art.png"/>
    <span><strong>Fig 1.1.</strong> Agent and Environment interactions</span>
</div>

-- Links --
[(k-fold) Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)

{% highlight python %}
@jit
def f(x)
    print("hi")
# does cool stuff
{% endhighlight %}

-- Highlights --
AAABC `ASDF` __some bold text__

-- Colors --
The <strong style="color: #1E72E7">joint distribution</strong> of $X$ and $Y$ is written as $P(X, Y)$.
The <strong style="color: #ED412D">marginal distribution</strong> on the other hand can be written out as a table.
-->

<p class="vspace"></p>

### Appendix: Statistical Basics (Variance, Covariance, Covariance matrix)

__Variance__ (scale by N-1 for a population): 

$$var(x) = \frac{1}{N} \sum_{i=1}^{N} (X_i - \mu_X)^2$$

__Covariance__ (scale by N-1 for a population): 

$$cov(X,Y) = \frac{1}{N} \sum_{i=1}^{N} (X_i - \mu_X) (Y_i - \mu_Y)$$

If you plug in X two times into the covariance formula, notice that $cov(X,X) = var(X)$.


### References
1. [Thumbnail: Gaussian process regression (sklearn docs)](https://scikit-learn.org/0.24/auto_examples/gaussian_process/plot_gpr_noisy_targets.html)
2. [Baysian Optimization - Implementation for Hyperparameter Tuning](https://github.com/fmfn/BayesianOptimization/blob/master/examples/basic-tour.ipynb) (GitHub Repo)
3. [Nando de Freitas: Machine learning - Introduction to Gaussian processes](https://www.youtube.com/watch?v=4vGiHC35j9s)
4. [Cornell: Lecture 15: Gaussian Processes](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote15.html) (well written notes)
5. [Gaussianprocess Notes](http://gaussianprocess.org/gpml/chapters/RW2.pdf) (includes implementation)
6. [Peter Roelants's blog: Gaussian processes (1/3) - From scratch ](https://peterroelants.github.io/posts/gaussian-process-tutorial/)
7. [Martin Krasser's blog: Bayesian optimization](http://krasserm.github.io/2018/03/21/bayesian-optimization/) (includes Expected Improvement formula)


<!-- Ressources -->
[transformer-img]: https://deepfrench.gitlab.io/deep-learning-project/resources/transformer.png


<!-- Optional Comment Section-->
{% if page.comments %}
<p class="vspace"></p>
<a class="commentlink" role="button" href="/comments/">Post a comment.</a> <!-- role="button"  -->
{% endif %}

<!-- Optional Back to Top Button -->
{% if page.back_to_top_button %}
<script src="https://unpkg.com/vanilla-back-to-top@7.2.1/dist/vanilla-back-to-top.min.js"></script>
<script>addBackToTop({
  diameter: 40,
  backgroundColor: 'rgb(255, 255, 255, 0.7)', /* 30,144,255, 0.7 */
  textColor: '#4a4946'
})</script>
{% endif %}