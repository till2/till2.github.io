---
layout: post
title:  "Actor Critic (👷)"
author: "Till Zemann"
date:   2022-10-24 20:36:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
positive_reward: true
reward: 2
tags: [reinforcement learning]
thumbnail: "/images/robot-2.png"
---

<div class="img-block" style="width: 300px;">
    <img src="/images/robot-2.png"/>
</div>


### Contents
* TOC
{:toc}

### Temporal Difference (TD) Error

We can calculate the TD error as the difference between the new and old estimates of a state value:

<strong style="color: #ED412D">$\delta^{\pi_{\theta}} = r + \gamma V^{\pi_{\theta}}(s') - V^{\pi_{\theta}}(s)$</strong>.

The TD Error <strong style="color: #ED412D">$\delta^{\pi_{\theta}}$</strong> is an unbiased estimate for the advantage <strong style="color: #ED412D">$A^{\pi_{\theta}(s,a)}$</strong>, meaning $\mathbb{E}[\delta^{\pi_{\theta}}] = A^{\pi_{\theta}(s,a)}$. This property will be helpful later.


### Policy gradient theorem

$$\nabla_{\theta} J(\theta) = \mathbb{E}[\nabla_{\theta} \log \pi_{\theta}(s,a)] R^s_a = \mathbb{E}[\nabla_{\theta} \log \underbrace{\pi_{\theta}(s,a)}_\text{actor} ] \overbrace{Q^{\pi_{\theta}}(s,a)}^\text{critic}$$.

$R^s_a$ is the expected reward signal that the agent receives taking action $a$ in state $s$.

### What are Actor and Critic?

In Deep Reinforcement Learning, an actor refers to the policy network while the critic represents a value network that is used to calculate either state-action values <strong style="color: #1E72E7">$Q^{\pi_{\theta}}(s,a)$</strong>, state values <strong style="color: #1E72E7">$V^{\pi_{\theta}}(s)$</strong> or an advantage value <strong style="color: #ED412D">$A^{\pi_{\theta}}(s,a)$</strong>. We will take a closer look at the advantage value that is used in the `Advantage Actor Critic (A2C)` algorithm. The advantage intuitively describes how much better or worse an action's value is compared to the current state value (How much advantage can i gain from taking this action in comparison to other actions?) and can be calculated using the advantage function as follows: <strong style="color: #ED412D">$A^{\pi_{\theta}}(s,a) = Q^{\pi_{\theta}}(s,a) - V^{\pi_{\theta}}(s)$</strong>

I guess the advantage will converge to be sligtly negative or zero with time as <strong style="color: #1E72E7">$V^{\pi_{\theta}}(s)$</strong> should converge towards <strong style="color: #1E72E7">$\max_a Q^{\pi_{\theta}}(s,a)$</strong>. For an optimal policy it should be zero. `[Not tested yet.]`


<em>Note, that it is common to use a shared neural network body. This is practical for learning features only once and not individually for both networks. The last layer of the body network connected to both the `policy head` and the `value head`), representing the actor and critic respectively, as follows:</em>

<div class="img-block" style="width: 500px;">
    <img src="https://www.datahubbs.com/wp-content/uploads/2018/08/two_headed_network.png"/>
</div>

### Gradient of the objective function

The objective function $J(\theta)$ gives us the future return. We want to find parameters $\theta$ that maximize $J(\theta)$ by gradient ascent. For that, we need the gradient of the objective function $J(\theta)$ w.r.t. $\theta$.

$\nabla_{\theta} J(\theta) = \mathbb{E}[\nabla_{\theta} \log \pi_{\theta}(s,a)] * A^{\pi_{\theta}}(s,a)$


### Actor Critic Algorithm

<div class="img-block" style="width: 500px;">
    <img src="/images/actionvalue-actor-critic-code.png"/>
</div>

### Todo
- Advantage Actor Critic (A2C) Algorithm
- clean implementation
- Evaluation and tradeoffs

Look at:
- [towardsdatascience blogpost](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f)
- [berkeley lecture slides](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf)
- [CS885 Lecture 7b: Actor Critic](https://www.youtube.com/watch?v=5Ke-d1Itk3k)
- [Actor Critic blogpost][actor-critic-blogpost]
- [TD0 Actor Critic code][actor-critic-TD0-code]

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


<!-- Ressources -->
[myreference-1]: https://www.youtube.com/watch?v=dQw4w9WgXcQ

### References
1. Picture taken from [here][datahubbs-pic-link].
2. Nice ressource on A2C (1-step and n-step) with code [here][datahubbs-a2c].
3. Pseudocode Image taken from [here][code].
4. PyTorch Actor Critic [implementation][torch-actor-critic-code].
5. TD0 Actor Critic [implementation][actor-critic-TD0-code]

<!-- Ressources -->
[datahubbs-pic-link]: https://www.datahubbs.com/two-headed-a2c-network-in-pytorch/
[datahubbs-a2c]: https://www.datahubbs.com/policy-gradients-and-advantage-actor-critic/
[code]: https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
[torch-actor-critic-code]: https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
[actor-critic-TD0-code]: https://github.com/chengxi600/RLStuff/blob/master/Actor-Critic/Actor-Critic_TD_0.ipynb
[actor-critic-blogpost]: https://medium.com/geekculture/actor-critic-value-function-approximations-b8c118dbf723


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