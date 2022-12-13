---
layout: post
title:  "Actor-Critics (👷)"
author: "Till Zemann"
date:   2022-12-13 00:32:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
positive_reward: true
reward: 2
tags: [reinforcement learning]
thumbnail: "/images/robot-2.png"
---

<em>First draft: 2022-10-24</em><br>
<em>Major rewrite and additions on: 2022-12-12</em>

<!-- add the actor-critic diagram from Prof. Sutton.! -->

<div class="img-block" style="width: 300px;">
    <img src="/images/robot-2.png"/>
</div>


### Contents
* TOC
{:toc}

### Introduction

Let's start by looking at the REINFORCE algorithm, a method for training reinforcement learning (RL) agents. It is a policy gradient method, which means that it uses gradient ascent to adjust the parameters of the policy in order to maximize the expected reward. It does this by computing the gradient of the performance (goal) $J(\theta) \stackrel{.}{=} V^{\pi_\theta}(s_0)$ with respect to the policy parameters, and then updating the policy in the direction of this gradient. This update rule is known as the policy gradient update rule, and it ensures that the policy is always moving in the direction that will increase the expected future reward (=return). Because we need the entire return $G_t$ for the update at timestep $t$, REINFORCE is a Monte-Carlo method and theirfore only well-defined for episodic cases.

By using a baseline $b(S_t)$, we can reduce the variance of the gradients and improve the stability of the learning process. The Actor-Critic algorithm is an extension of the REINFORCE algorithm that uses a value function as a baseline to improve the stability of the learning process. This baseline also needs to be learned (we have to approximate $V(s)$, usually using a Deep Neural Network), theirfore Actor-Critics are a combination of value-based and policy-based methods.

<div class="img-block" style="width: 400px;">
    <img src="/images/actor-critic/venn-simple.jpg"/>
</div>

The image below depicts a more comprehensive venn diagram for the RL-algorithm taxonomy (which also distinguish between model-based and model-free algorithms).
<div class="img-block" style="width: 700px">
    <img src="/images/actor-critic/venn-diagram-rl-algos-detailed.png"/>
</div>


### From Policy-Gradient-Theorem to REINFORCE update rule


The policy gradient theorem (for the episodic case) states that:

$$
\begin{align*}
\nabla_{\theta} J(\theta) &\propto \sum_s \mu(s) \sum_a Q^\pi(s,a) \nabla \pi(a|s,\theta) \\
                          &= \mathbb{E_\pi}[ \sum_a Q^\pi(s,a) \nabla \pi(a|s,\theta) ]
\end{align*}
$$

where $\mu(s)$ is the on-policy distribution over all states (included in $\mathbb{E}_\pi$). (From S&B [[6]][sab], Chapter 13). \\
We can extend it further by <strong style="color: #ED412D">multiplying and deviding by $\pi(a|S_t, \theta)$ to get the expression $\frac{\nabla \pi(a|S_t, \theta)}{\pi(a|S_t, \theta)}$</strong>. This is a common trick using the logarithm, where you can rewrite the gradient of $\log x$ with $\nabla \log x = \frac{1}{x} \nabla x = \frac{\nabla x}{x}$ (just using the chain rule). In our specific case, we can use this as  <strong style="color: #1E72E7">$\nabla \log \pi(a|S_t, \theta)$</strong> $=$ <strong style="color: #ED412D">$\frac{\nabla \pi(a|S_t, \theta)}{\pi(a|S_t, \theta)}$</strong>. 

Performing all of the steps above:

$$
\nabla_{\theta} J(\theta)   \propto \mathbb{E_\pi}[ \sum_a Q^\pi(s,a) \nabla \pi(a|s,\theta)]
$$

<center>
<!-- with multiplied and devided by pi(a|S_t,\theta) -->
$
= \mathbb{E_\pi}[ \sum_a 
$
<strong style="color: #ED412D">$\pi(a|S_t, \theta)$</strong>
$Q^\pi(s,a)$
<strong style="color: #ED412D">$\frac{\nabla \pi(a|S_t, \theta)}{\pi(a|S_t, \theta)}$</strong>$]$ <br><br>
</center>

We can just replace 
$
\sum_a 
$
<strong style="color: #ED412D">$\pi(a|S_t, \theta)$</strong>
$= 1$ and use the log-trick (rewrite the gradient of $\log x$ as the fraction described in the section above).

<!-- rewritten as gradient of log -->
<center>
$
= \mathbb{E_\pi}[Q^\pi(s,a)
$
<strong style="color: #1E72E7">$\nabla \log \pi(a|S_t, \theta)$</strong>$]$
<br><br>
$
= \mathbb{E_\pi}[G_t
$
<strong style="color: #1E72E7">$\nabla \log \pi(a|S_t, \theta)$</strong>$]$

</center>
<br>

<!-- Gradient for the actor critic -->
<!--
$$\nabla_{\theta} J(\theta) = \mathbb{E}[\nabla_{\theta} \log \pi_{\theta}(s,a)] R^s_a = \mathbb{E}[\nabla_{\theta} \log \underbrace{\pi_{\theta}(s,a)}_\text{actor} ] \overbrace{Q^{\pi_{\theta}}(s,a)}^\text{critic}$$.

$R^s_a$ is the expected reward signal that the agent receives taking action $a$ in state $s$.
-->

### Temporal Difference (TD) Error

We can calculate the TD error as the difference between the new and old estimates of a state value:

<strong style="color: #ED412D">$\delta^{\pi_{\theta}} = r + \gamma V^{\pi_{\theta}}(s') - V^{\pi_{\theta}}(s)$</strong>.

The TD Error <strong style="color: #ED412D">$\delta^{\pi_{\theta}}$</strong> is an unbiased estimate for the advantage <strong style="color: #ED412D">$A^{\pi_{\theta}(s,a)}$</strong>, meaning $\mathbb{E}[\delta^{\pi_{\theta}}] = A^{\pi_{\theta}(s,a)}$. This property will be helpful later.



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
- [Richard Sutton: Actor-Critic Methods](http://incompleteideas.net/book/ebook/node66.html) !!! The inventoooor.
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
5. TD(0) Actor Critic [implementation][actor-critic-TD0-code]
6. [Sutton & Barto: Reinforcement Learning, An introduction (second edition)][sab]


<!-- Ressources -->
[datahubbs-pic-link]: https://www.datahubbs.com/two-headed-a2c-network-in-pytorch/
[datahubbs-a2c]: https://www.datahubbs.com/policy-gradients-and-advantage-actor-critic/
[code]: https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
[torch-actor-critic-code]: https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
[actor-critic-TD0-code]: https://github.com/chengxi600/RLStuff/blob/master/Actor-Critic/Actor-Critic_TD_0.ipynb
[actor-critic-blogpost]: https://medium.com/geekculture/actor-critic-value-function-approximations-b8c118dbf723
[sab]: http://incompleteideas.net/book/the-book-2nd.html

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