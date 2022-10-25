---
layout: post
title:  "Actor Critic"
author: "Till Zemann"
date:   2022-10-24 20:36:41 +0200
categories: jekyll update
math: true
---

* TOC
{:toc}

## Temporal Difference (TD) Error

We can calculate the TD error as the difference between the new and old estimates of a state value:

<strong style="color: #ED412D">$\delta^{\pi_{\theta}} = r + \gamma V^{\pi_{\theta}}(s') - V^{\pi_{\theta}}(s)$</strong>.

The TD Error <strong style="color: #ED412D">$\delta^{\pi_{\theta}}$</strong> is an unbiased estimate for the advantage <strong style="color: #ED412D">$A^{\pi_{\theta}(s,a)}$</strong>, meaning $\mathbb{E}[\delta^{\pi_{\theta}}] = A^{\pi_{\theta}(s,a)}$. This property will be helpful later.


## Policy gradient theorem

Short and sweet, here it is.

$\nabla_{\theta} J(\theta) = \mathbb{E}[\nabla_{\theta} \log \pi_{\theta}(s,a)] R^s_a = \mathbb{E}[\nabla_{\theta} \log \pi_{\theta}(s,a)] Q^{\pi_{\theta}}(s,a)$.

$R^s_a$ is the step reward for taking action $a$ in state $s$.

## What are Actor and Critic?

In Deep Reinforcement Learning, an actor refers to the policy network while the critic represents a value network that is used to calculate either state-action values <strong style="color: #1E72E7">$Q^{\pi_{\theta}}(s,a)$</strong>, state values <strong style="color: #1E72E7">$V^{\pi_{\theta}}(s)$</strong> or an advantage value <strong style="color: #ED412D">$A^{\pi_{\theta}}(s,a)$</strong>. We will take a closer look at the advantage value that is used in the `Advantage Actor Critic (A2C)` algorithm. The advantage intuitively describes how much better or worse an action's value is compared to the current state value (How much advantage can i gain from taking this action in comparison to other actions?) and can be calculated using the advantage function as follows: <strong style="color: #ED412D">$A^{\pi_{\theta}}(s,a) = Q^{\pi_{\theta}}(s,a) - V^{\pi_{\theta}}(s)$</strong>

I guess the advantage will converge to be sligtly negative or zero with time as <strong style="color: #1E72E7">$V^{\pi_{\theta}}(s)$</strong> should converge towards <strong style="color: #1E72E7">$\max_a Q^{\pi_{\theta}}(s,a)$</strong>. For an optimal policy it should be zero. `[Not tested yet.]`


Note, that it is common to use a shared body neural net that learns useful features from the input and then put two seperate networks (called `policy head` and `value head`) representing the actor and critic on top as follows:

<div class="img-block" style="width: 500px;">
    <img src="https://www.datahubbs.com/wp-content/uploads/2018/08/two_headed_network.png"/>
    <span><strong>Fig. 1: </strong>Actor Critic network with body</span>
</div>

## Gradient of the objective function

The objective function $J(\theta)$ gives us the future return. We want to find parameters $\theta$ that maximize $J(\theta)$ by gradient ascent. For that, we need the gradient of the objective function $J(\theta)$ w.r.t. $\theta$.

$\nabla_{\theta} J(\theta) = \mathbb{E}[\nabla_{\theta} \log \pi_{\theta}(s,a)] * A^{\pi_{\theta}}(s,a)$


## Advantage Actor Critic (A2C) Algorithm



## Evaluation and tradeoffs



<!-- Code Box -->
{% highlight python %}
@jit
def f(x)
    print("hi")
# does cool stuff
{% endhighlight %}


<!-- In-Text Citing -->
<!-- 
You can...
- use bullet points
1. use
2. ordered
3. lists


do $X$ math

embed images:
<div class="img-block" style="width: 800px;">
    <img src="/images/lofi_art.png"/>
    <span><strong>Fig 1.1.</strong> Agent and Environment interactions</span>
</div>

refer to links:
[(k-fold) Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)

{% highlight python %}
@jit
def f(x)
    print("hi")
# does cool stuff
{% endhighlight %}
-->


<!-- Ressources -->
[myreference-1]: https://www.youtube.com/watch?v=dQw4w9WgXcQ

<!-- Normal Text and Highlights -->
AAABC `ASDF` __some bold text__

<!-- Text with Colors -->
The <strong style="color: #1E72E7">joint distribution</strong> of $X$ and $Y$ is written as $P(X, Y)$.
The <strong style="color: #ED412D">marginal distribution</strong> on the other hand can be written out as a table.

<!-- Math Text -->
We can write a formula into text: $V(S_t) \gets V(S_t) + \alpha [ V(S_{t+1}) - V(S_t) ]$.


## References
1. Picture taken from [here][datahubbs-pic-link].
2. Nice ressource on A2C (1-step and n-step) with code [here][datahubbs-a2c]

<!-- Ressources -->
[datahubbs-pic-link]: https://www.datahubbs.com/two-headed-a2c-network-in-pytorch/
[datahubbs-a2c]: https://www.datahubbs.com/policy-gradients-and-advantage-actor-critic/