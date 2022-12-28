---
layout: post
title:  "Trust region methods"
author: "Till Zemann"
date:   2022-12-22 16:31:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
positive_reward: true
reward: 2
tags: [reinforcement learning, 🟢 not finished]
thumbnail: "/images/trust_region_methods/0 DAKbTuPaiGXOUd_e.webp"
---

<!-- add the actor-critic diagram from Prof. Sutton.! -->

<div class="img-block" style="width: 600px;">
    <img src="/images/trust_region_methods/0 DAKbTuPaiGXOUd_e.webp"/>
</div>
<center>Image taken from <a href="https://medium.com/analytics-vidhya/trust-region-methods-for-deep-reinforcement-learning-e7e2a8460284">[1]</a>.</center>


<!-- <em style="float:right">First draft: 2022-10-22</em><br> -->

<!--
### Contents
* TOC
{:toc}
-->

### Introduction

- Trust region methods: TRPO and PPO
- bad policy leads to bad data
- TRPO needs 2nd order derivative, hard to implement (?) -> just use PPO
- PPO has KL and Clip variants
- KL divergence approximation trick (see kl, some blogpost)
- Clip is easier to implement
- Show clip plots



### Derivation of the Surrogate Loss Function

We are starting with the policy gradient (more details in my <a href="/blog/2022/12/20/actorcritic">Actor Critic blogpost</a>), that we want to maximize. Next, we use the log-trick (for more details I also refer to the Actor Critic blogpost), which gets us the policy ratio multiplied with the advantage (I'm not sure why we can use an old policy and get the correct expectation here; researching this further...):

$$
\begin{align*}
\nabla J(\theta)	&= \mathbb{E}_{\pi_{\theta}} \left[ \nabla \log \pi_{\theta}(a|s) A_{\text{w}}(s,a) \right] \\
					&=^{why?} \mathbb{E}_{\pi_{\theta}} \left[ \frac{\nabla \pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A_{\text{w}}(s,a) \right] \\
\end{align*}
$$

Note that the `importance sampling ratio` (the first fraction) is also often written abbreviated, for example as $r(\theta) \dot{=} \frac{\pi_{\theta} (a\|s)}{\pi_{\theta_\text{old}} (a\|s)}$ ($r$ stands for ratio).


Now we can extract a loss function (that gets minimized!):

$$
\Rightarrow \mathcal{L}_{\text{actor}} = - \frac{\nabla \pi_{\theta}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A_{\text{w}}(s,a)
$$


<!--
$$
L_{\text{surrogate}} = \text{clip}(\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)},1-\epsilon,1+\epsilon) \sum_{s \in S} \sum_{a \in A} \pi_\theta(a|s) \log \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}
$$

This constraint is controlled by a hyperparameter called epsilon, which determines the maximum allowed difference between the current and previous policies.
-->



### TRPO


### PPO


### TODO

- first look at [importance sampling](https://youtu.be/C3p2wI4RAi8)
- then watch Pieters lecture and take notes
- then research anything that's unclear
- implement it

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
1. Thumbnail taken from [here][trust-region-methods-blogpost].
2. [Pieter Abbeel: L4 TRPO and PPO (Foundations of Deep RL Series) ][pieter-abbeel-trpo-ppo-lecture]


<!-- Ressources -->
[trust-region-methods-blogpost]: https://medium.com/analytics-vidhya/trust-region-methods-for-deep-reinforcement-learning-e7e2a8460284
[pieter-abbeel-trpo-ppo-lecture]: https://www.youtube.com/watch?v=KjWF8VIMGiY&list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0&index=4

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
