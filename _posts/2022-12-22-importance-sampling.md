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
tags: [reinforcement learning, stochastics, 🟢 not finished]
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

Importance Sampling is a tool for estimating the expectation of a random variable that we can not sample from. It is applicable when the following preconditions are met:

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
                    &= \sum_x \frac{ g(x) }{ f(x) } x f(x)          \;\; (\text{multiply with} \frac{f(x)}{f(x)}) \\
                    &= \mathbb{E}_f \left[ \frac{g(x) }{ f(x) } x \right] \\
\end{align*}
$$

Now we have rewritten the expectation into an expectation with respect to $f$, the _behavior (sampling) policy_. We can estimate this expectation as usual:

$$
\mathbb{E}_f \left[ \frac{g(x) }{ f(x) } x \right] \approx \frac{1}{n} \sum_{i=1}{n} \frac{g(x_i)}{f(x_i)} x
$$

<!-- MIGHT BE WRONG!

### Application to off-policy policy gradient methods

- application for example in PPO (to update from a minibatch of samples) and Actor Critics with Experience Replay (ACER)

Using importance sampling to estimate the policy gradient:

$$
\begin{align*}
\nabla J(\pi_\text{target})  &= \mathbb{E}_{\pi_{\text{target}}} \left[ \nabla \log \pi_\text{target}(a|s) A_\text{target}(s,a) \right] \\
&= \mathbb{E}_{\pi_{\text{behavior}}} \left[ \frac{\pi_\text{target}(a|s)}{\pi_\text{behavior}(a|s)} \nabla \log \pi_\text{behavior}(a|s) A_\text{behavior}(s,a) \right]
\end{align*}
$$

Note that the `importance sampling ratio` (the first fraction) is also often written abbreviated, for example as $r(\theta) \dot{=} \frac{\pi_{\theta} (a\|s)}{\pi_{\theta_\text{old}} (a\|s)}$ where $r$ stands for ratio.
-->


### TODO

- check if math is correct
- implement an example for a coin toss or better yet a die (one with fair and one with unfair probs)
- copy the example from "Phil Winder, Ph.D. -- Reinforcement Learning: Industrial Applications of Intelligent Agents" to check if my solution is correct.
- https://rl-book.com/learn/statistics/importance_sampling/

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
2. Thumbnail taken from [Sequential importance sampling for low-probability and high-dimensional SRAM yield analysis][thumbnail].

<!-- Ressources -->
[thumbnail]: https://www.semanticscholar.org/paper/Sequential-importance-sampling-for-low-probability-Katayama-Hagiwara/7e8ad118a0c1de96d29147aa58518a2ca161c48e


<!-- Optional Comment Section-->
{% if page.comments %}
<p class="vspace"></p>
<a class="commentlink" role="button" href="/comments/">Share your thoughts.</a> <!-- role="button"  -->
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