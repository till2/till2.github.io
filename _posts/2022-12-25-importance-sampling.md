---
layout: post
title:  "Importance Sampling"
author: "Till Zemann"
date:   2022-12-22 16:31:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
positive_reward: true
reward: 2
tags: [math]
thumbnail: "/images/trpo-ppo/thumbnail.jpeg"
---

<!--
<div class="img-block" style="width: 300px;">
    <img src="/images/trpo-ppo/thumbnail.jpeg"/>
</div>
-->

<!-- <em style="float:right">First draft: 2022-10-22</em><br> -->

<!--
### Contents
* TOC
{:toc}
-->

### Introduction

Importance Sampling is a tool for when we have two random variables with the same values (e.g. the values of a die or values for policy gradient estimate samples) and you want to get an expectation with respect to a probability function which you can not sample from (e.g. because it is too expensive in the case of policy gradient estimate samples), but you know the probabiilies of the individual values.
With importance sampling, we can rewrite the expectation w.r.t. the target random variable into an expectation w.r.t. the random variable that we can sample from.

With this trick, we can use offline RL (replay buffers) for policy gradient methods! This is only possible because we can rewrite the expectation w.r.t. the policy gradient of a target policy into the expectation w.r.t. a behavior policy, which might be an old policy that was used to collect samples which are now stored in a replay buffer. It could also be samples collected from e.g. humans as expert samples for a given task.

### Derivation

Definition for an expectation $\mathbb{E}$ of a random variable $x$ with regard to a probability function f (and g in the second case):

$$
\begin{align*}
\mathbb{E}_f [x]    &\dot{=} \sum_x f(x) x \\
\mathbb{E}_g [x]    &\dot{=} \sum_x g(x) x
\end{align*}
$$

$$
\begin{align*}
\mathbb{E}_g [x]    &\dot{=} \sum_x g(x) x \\
                    &= \sum_x \frac{ g(x) x }{ f(x) } f(x)          \;\; (\text{multiply with} \frac{f(x)}{f(x)} = 1 ) \\
                    &= \mathbb{E}_f [ \frac{g(x) x }{ f(x) } ] \\
\end{align*}
$$

Now we have rewritten the expectation into an expectation with respect to $f$, i.e. the _behavior (sampling) policy_. We can estimate this expectation as usual:

$$
\mathbb{E}_f [ \frac{g(x) }{ f(x) } x ] \approx \frac{1}{n} \sum_{i=1}{n} \frac{g(x_i)}{f(x_i)} x \\
$$

### Application to off-policy policy gradient methods <br> (i.e. Actor Critics with a replay buffer)

Using importance sampling to estimate the policy gradient:

$$
\begin{align*}
\nabla J(\pi_\text{target})  &= \mathbb{E}_\pi_\text{target} [ \nabla \log \pi_\text{target}(a|s) A_\text{target}(s,a)] \\
&= \mathbb{E}_\pi_\text{behavior} [ \frac{\pi_\text{target}(a|s)}{\pi_\text{behavior}(a|s)} \nabla \log \pi_\text{behavior}(a|s) A_\text{behavior}(s,a)]
\end{align*}
$$

Note that the $importance sampling ratio$ (the first fraction) is also often written abbreviated, for example as $r(\theta) \dot{=} \frac{\pi_\theta (a|s)}{\pi_\theta_\text{old} (a|s)}$ where $r$ stands for ratio.

### TODO

- implement an example for a coin toss or better yet a die (one with fair and one with unfair probs)
- copy the example from "Phil Winder, Ph.D. -- Reinforcement Learning: Industrial Applications of Intelligent Agents" to check if my solution is correct.


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

<!-- Ressources -->
[thumbnail]: https://arxiv.org/pdf/2007.04309.pdf


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