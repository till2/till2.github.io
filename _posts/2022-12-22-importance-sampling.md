---
layout: post
title:  "Importance Sampling"
author: "Till Zemann"
date:   2022-12-22 08:31:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
positive_reward: true
reward: 2
tags: [reinforcement learning, stochastics]
thumbnail: "/images/importance_sampling/thumbnail.png"
---


<div class="img-block" style="width: 500px;">
    <img src="/images/importance_sampling/thumbnail.png"/>
</div>
<center>Image taken from <a href="https://www.semanticscholar.org/paper/Sequential-importance-sampling-for-low-probability-Katayama-Hagiwara/7e8ad118a0c1de96d29147aa58518a2ca161c48e">[2]</a></center>


<!-- <em style="float:right">First draft: 2022-10-22</em><br> -->

<!--
### Contents
* TOC
{:toc}
-->

### Introduction

This blogpost describes the mathematical background for off-policy policy gradient methods.

Importance sampling is a statistics tool for estimating the expectation of a random variable that we can not sample from. It is applicable when the following preconditions are met:

- we can't sample from the random variable of interest, maybe because it's too expensive <br> (otherwise just sample from it)
- but we know the probabilities of the individual values of this random variable
- we have a second random variabe that can take on the same values and that we can sample from.

This is especially useful in Reinforcement Learning if we already have some collected samples from an old policy and we want to update a new policy with them. The goal is to calculate the expectation of the new policy, and all the prerequisites are satisfied because for each collected experience, we can calculate the probability that the new policy would have taken that action.

With importance sampling, we can rewrite the expectation of the target random variable into an expectation with respect to the second random variable that we can sample from.

Using this trick, we can use offline Reinforcement Learning (replay buffers) for policy gradient methods!


### Derivation

Definitions for expectations of a random variable $x$ with regard to a probability function $f$ (and in the second case $g$):

$$
\begin{align*}
\mathbb{E}_f [x]    &\dot{=} \sum_x f(x) x \\
\mathbb{E}_g [x]    &\dot{=} \sum_x g(x) x
\end{align*}
$$


Rewriting the expectation with respect to another function (from target $g$ to sampling variable $f$):

$$
\begin{align*}
\mathbb{E}_g [x]    &\dot{=} \sum_x g(x) x \\
                    &= \sum_x f(x) \frac{ g(x) }{ f(x) } x          \;\; (\text{multiply with} \frac{f(x)}{f(x)}) \\
                    &= \mathbb{E}_f \left[ \frac{g(x) }{ f(x) } x \right] \\ \\
\end{align*}
$$

This results in the importance sampling formula.

$$
\Rightarrow \mathbb{E}_f \left[ \frac{g(x) }{ f(x) } x \right] = \mathbb{E}_g \left[ x \right]
$$

Now that we have rewritten the expectation with respect to $f$ (_the behavior/ sampling policy_), we can just estimate this expectation using $N$ samples:

$$
\mathbb{E}_f \left[ \frac{g(x) }{ f(x) } x \right] \approx \frac{1}{N} \sum_{i=1}^{N} \frac{g(x_i)}{f(x_i)} x_i
$$

<p class="vspace"></p>


### Implementation

I used the example (the two dice with the given probabilities) from [2] so that I'm able to validate my results.

First we define our values, which are the face values of a normal 6-faced die.

```py
import numpy as np

# values
x = np.arange(1,7) # 1-6
```

Then we define the probability functions $f$ and $g$.

```py
# define f
f_probs = np.array([1 / len(x) for x_i in x]) # uniform (fair) distribution
print(f_probs)
```

<div class="output">
array([0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
       0.16666667])
</div>

```py
# define g
g_probs = np.array([x_i / len(x) for x_i in x]) # biased (unfair) distribution
g_probs /= sum(g_probs)
print(g_probs)
```

<div class="output">
array([0.04761905, 0.0952381 , 0.14285714, 0.19047619, 0.23809524,
       0.28571429])
</div>

Let's start by approximating the mean of a fair die with $N=5000$ samples.

$$
\mathbb{E}_f [x] \approx \frac{1}{N} \sum_{i=1}^{N} x_i
$$

```py
N_samples = 5000

# get the samples using a pseudo-random generator
fair_die_samples = np.random.choice(x, size=N_samples, p=f_probs)

# calculate the mean of the samples for the fair die
approx_fair_die_mean = 1 / N_samples * np.sum(fair_die_samples)

# print the mean
print(f'approx_fair_die_mean: {approx_fair_die_mean:.3f}')
```

<div class="output">
approx_fair_die_mean: 3.501
</div>

Now comes the importance sampling part to estimate $\mathbb{E}_g[x]$ by sampling from $f$.

$$
\mathbb{E}_g \left[ x \right] = \mathbb{E}_f \left[ \frac{g(x) }{ f(x) } x \right] \approx \frac{1}{N} \sum_{i=1}^{N} \frac{g(x_i)}{f(x_i)} x_i
$$

```py
# calculate the mean of the samples for the biased die using importance sampling
approx_biased_die_mean = 0.0

for i in range(N_samples):
    x_i = np.random.choice(x, p=f_probs)
    approx_biased_die_mean += (g_probs[x_i-1] / f_probs[x_i-1]) * x_i
    
approx_biased_die_mean /= N_samples

# print the mean
print(f'approx_biased_die_mean: {approx_biased_die_mean:.3f}')
```

<div class="output">
approx_biased_die_mean: 4.336
</div>


For reference, the theoretical expected values are:

$$
\mathbb{E}_f [x] = 3.5 \\
\mathbb{E}_g [x] = 4.\bar{3}
$$

<p class="vspace"></p>


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

<!-- uncomment, when i understand more of the algorithms presented (missing DDPG, SAC, TD3, TRPO, PPO, Dyna-Q)
### Rl-Algorithms-Taxonomy in a Venn-Diagram

<div class="img-block" style="width: 700px;">
    <img src="/images/actor-critic/venn-diagram-rl-algos-detailed.png"/>
</div>

-->

### References

1. Phil Winder, Ph.D. - Reinforcement Learning: Industrial Applications of Intelligent Agents
2. [Phil Winder - Importance Sampling Notebook][phil-winder-notebook]
3. Thumbnail taken from [Sequential importance sampling for low-probability and high-dimensional SRAM yield analysis][thumbnail].

<!-- Ressources -->
[thumbnail]: https://www.semanticscholar.org/paper/Sequential-importance-sampling-for-low-probability-Katayama-Hagiwara/7e8ad118a0c1de96d29147aa58518a2ca161c48e
[phil-winder-notebook]: https://rl-book.com/learn/statistics/importance_sampling/

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