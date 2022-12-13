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

<!-- add the actor-critic diagram from Prof. Sutton.! -->

<div class="img-block" style="width: 300px;">
    <img src="/images/robot-2.png"/>
</div>

<em style="float:right">First draft: 2022-10-24</em><br>
<em style="float:right">Second draft: 2022-12-12</em>

<!--
### Contents
* TOC
{:toc}
-->

### Introduction

Let's start by looking at the REINFORCE algorithm, a method for training reinforcement learning (RL) agents. It is a policy gradient method, which means that it uses gradient ascent to adjust the parameters of the policy in order to maximize the expected reward. It does this by computing the gradient of the performance (goal) $J(\theta) \stackrel{.}{=} V^{\pi_\theta}(s_0)$ with respect to the policy parameters, and then updating the policy in the direction of this gradient. This update rule is known as the policy gradient update rule, and it ensures that the policy is always moving in the direction that will increase the expected future reward (=return). Because we need the entire return $G_t$ for the update at timestep $t$, REINFORCE is a Monte-Carlo method and theirfore only well-defined for episodic cases. 

One drawback of the pure REINFORCE algorithm is, that it has a really high variance and could be unstable as a result. The baseline $b(S_t)$ has to be independent of the action. A good idea is to use state-values as a baseline, which reduce the magnitude of the expected reward (it has the effect of "shrinking" the estimated rewards towards the baseline value). Reducing the magnitude of the estimated rewards can help to reduce the variance of the algorithm. This is because the updates that the algorithm makes to the policy are based on the estimated rewards. If the magnitude of the rewards is large, the updates will also be large, which can cause the learning process to be unstable and can result in high variance. By reducing the magnitude of the rewards, the updates are also reduced, which can help to reduce the variance and thus stabilize the learning process.

The Actor-Critic algorithm is an extension of the REINFORCE algorithm that uses a value function as a baseline to improve the stability of the learning process. This baseline also needs to be learned (we have to approximate $V(s)$, usually using a Deep Neural Network), theirfore Actor-Critics are a combination of value-based and policy-based methods:


<!-- new -->
 <div class="row">
  <div class="column1">
    <img src="/images/actor-critic/venn-simple.jpg" style="width:100%">
  </div>
  <!--
  <div class="column2">
    <img src="/images/actor-critic/venn-diagram-rl-algos-detailed.png" style="width:100%">
  </div>
-->
</div> 
<br>

### From Policy-Gradient-Theorem to REINFORCE update rule


The policy gradient theorem (for the episodic case) states that:

$$
\begin{align*}
\nabla_{\theta} J(\theta) &\propto \sum_s \mu(s) \sum_a Q^\pi(s,a) \nabla \pi(a|s,\theta) \\
                          &= \mathbb{E_\pi}[ \sum_a Q^\pi(s,a) \nabla \pi(a|s,\theta) ]
\end{align*}
$$

where $\mu(s)$ is the on-policy distribution over all states (included in $\mathbb{E}_\pi$). (From S&B [[6]][sab], Chapter 13). \\
We can extend it further by multiplying and deviding by <strong style="color: #ED412D">$\pi(a|S_t, \theta)$</strong> to get the expression <strong style="color: #ED412D">$\frac{\nabla \pi(a|S_t, \theta)}{\pi(a|S_t, \theta)}$</strong>. This is a common trick using the logarithm, where you can rewrite the gradient of $\log x$ with $\nabla \log x = \frac{1}{x} \nabla x = \frac{\nabla x}{x}$ (just using the chain rule). In our specific case, we can use this as  <strong style="color: #1E72E7">$\nabla \log \pi(a|S_t, \theta)$</strong> $=$ <strong style="color: #ED412D">$\frac{\nabla \pi(a|S_t, \theta)}{\pi(a|S_t, \theta)}$</strong>. 

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

$$\delta = r + \gamma V(s') - V(s)$$

The TD-Error denotes how good or bad an action-value is compared to the average action-value and thus is an unbiased estimate of the advantage $A(s,a)$ of an action. This is helpful if we want to update our network after every transition, because we can just use use the TD-Error in the place of the advantage to approximate it. Proof that the TD-Error approximates the advantage:

$$
\begin{align*}
\mathbb{E}[\delta^\pi|s,a]  &= \mathbb{E}_\pi[G|s,a] - V^\pi(s,a) \\
                            &= Q^\pi(s,a) - V^\pi(s,a) \\
                            &= A^\pi(s,a)
\end{align*}
$$

### Variations 

1) If we want to get less variance and thus more stable updates, we could also calculate the advantage as the return $G_t \stackrel{.}{=} R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=t+1}^{\infty} \gamma^k R_{t+k+1}$ minus the state-value. For this variation of the actor-critic algorithm, we can only do updates after each episode (because we need to calculate the return $G_t$).

$$A(s_t,a) = G_t - V(s)$$

2) You could also estimate the advantage by using a critic Neural Network that estimates $V(s)$ and $Q(s,a)$ at the same time, and you just use $A(s,a) = Q(s,a) - V(s)$.

<div class="img-block" style="width: 500px;">
    <img src="/images/actor-critic/policy-gradient-variationen.png"/>
</div>



### What are Actor and Critic?

The main idea is that we update the actor parameters in the direction of the critic parameters. This makes sense because the critic is better able to evaluate the actual value of a state.

As already mentioned, the actor is responsible for learning a policy $\pi(a\|s)$, which is a function that determines the next action to take in a given state. The critic, on the other hand, is responsible for learning a value function $V(s)$ or $Q(s,a)$, which estimates the future rewards that can be obtained by following the policy. The actor and critic work together to improve the policy and value function over time, with the goal of maximizing the overall rewards obtained by the system.

<em>Note, that it is common to use a shared neural network body. This is practical for learning features only once and not individually for both networks. The last layer of the body network connected to both the `policy head` and the `value head`), representing the actor and critic respectively, as follows:</em>

<div class="img-block" style="width: 500px;">
    <img src="https://www.datahubbs.com/wp-content/uploads/2018/08/two_headed_network.png"/>
</div>


### Actor Critic Algorithm

<div class="img-block" style="width: 500px;">
    <img src="/images/actionvalue-actor-critic-code.png"/>
</div>

### Todo

- algorithm pseudocode
- actor-critic in our brain



- [Richard Sutton: Actor-Critic Methods](http://incompleteideas.net/book/ebook/node66.html) !!! The inventoooor.
- clean implementation

Look at:
- [towardsdatascience blogpost](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f)
- [berkeley lecture slides](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf)
- [CS885 Lecture 7b: Actor Critic](https://www.youtube.com/watch?v=5Ke-d1Itk3k)
- [Actor Critic blogpost][actor-critic-blogpost]
- [TD0 Actor Critic code][actor-critic-TD0-code]



### What i learned from writing this post

Reinforcement learning formalism can get really messy and unpleasent to look at, so it can sometimes get hard to absorb the important pieces of information. For this reason it is usually better to _omit some formalism and instead write clean looking formulas_ for the sake of readability, if the context of writing allows it (i.e. you are not writing a scientific paper). A piece that you can usually leave out if it is clear what we are referring to is $\theta$ in the subscript.





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


<!-- Ressources -->
[myreference-1]: https://www.youtube.com/watch?v=dQw4w9WgXcQ

### References
1. Illustration of the Neural Net architecture with a shared body taken from [here][datahubbs-pic-link].
2. Nice ressource on A2C (1-step and n-step) with code [here][datahubbs-a2c].
3. Pseudocode Image taken from [here][code].
4. PyTorch Actor Critic [implementation][torch-actor-critic-code].
5. TD(0) Actor Critic [implementation][actor-critic-TD0-code]
6. [Sutton & Barto: Reinforcement Learning, An introduction (second edition)][sab]
7. [Hado van Hasselt: Lecture 8 - Policy Gradient][hadovanhasselt]


<!-- Ressources -->
[datahubbs-pic-link]: https://www.datahubbs.com/two-headed-a2c-network-in-pytorch/
[datahubbs-a2c]: https://www.datahubbs.com/policy-gradients-and-advantage-actor-critic/
[code]: https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
[torch-actor-critic-code]: https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
[actor-critic-TD0-code]: https://github.com/chengxi600/RLStuff/blob/master/Actor-Critic/Actor-Critic_TD_0.ipynb
[actor-critic-blogpost]: https://medium.com/geekculture/actor-critic-value-function-approximations-b8c118dbf723
[sab]: http://incompleteideas.net/book/the-book-2nd.html
[hadovanhasselt]: https://hadovanhasselt.files.wordpress.com/2016/01/pg1.pdf

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