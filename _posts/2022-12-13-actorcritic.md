---
layout: post
title:  "Actor Critics"
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

### From the Policy-Gradient-Theorem to REINFORCE


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

Overview of the Actor-Critic variations:

<div class="img-block" style="width: 500px;">
    <img src="/images/actor-critic/policy-gradient-variationen.png"/>
</div>



### What are Actor and Critic?

The main idea is that we update the actor parameters in the direction of a value that is estimated by the critic, e.g. the advantage. This makes sense because the critic is better able to evaluate the actual value of a state.

<div class="img-block" style="width:350px;float:right;margin-left:20px">
    <img src="https://www.datahubbs.com/wp-content/uploads/2018/08/two_headed_network.png"/>
</div>

As already mentioned, the actor is responsible for learning a policy $\pi(a\|s)$, which is a function that determines the next action to take in a given state. The critic, on the other hand, is responsible for learning a value function $V(s)$ or $Q(s,a)$, which estimates the future rewards that can be obtained by following the policy. The actor and critic work together to improve the policy and value function over time, with the goal of maximizing the overall rewards obtained by the system.

<em>Note, that it is common to use a shared neural network body. This is practical for learning features only once and not individually for both networks. The last layer of the body network connected to both the `policy head` and the `value head`), representing the actor and critic respectively, as follows:</em>


### Actor Critic Algorithm

The following algorithm for an Actor Critic in the episodic case, we are calculating the TD-Error as $\delta \leftarrow R + \gamma \hat{V}(S',w) - \hat{V}(S,w)$, using our parameterized state-value function (the critic). This means that all bootstrappinging of the TD-Error depends on our current set of parameters, which can introduce a bias. Theirfore, the updates only include a part of the true gradient. These methods are called `semi-gradient` methods.


__Actor Critic algorithm (episodic):__

<hr>

__Input:__ 
policy parameterization $\pi(a|s,\theta)$  <em>(e.g. a Deep Neural Network)</em>,<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
state-value function parameterization $\hat{V}(s,\textbf{w})$ <em>(e.g. a Deep Neural Network)</em>,<br>

__Parameters:__ learning rates for the actor: $\alpha_\theta$, and for the critic :$\alpha_\textbf{w}$ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
discount-factor $\gamma$ 

0. Initialize the parameters in $\theta$ and $\textbf{w}$ arbitrarily (e.g. to 0) 
1. While True:<br>
    1. &nbsp; $S \leftarrow \text{env.reset()}$ &nbsp; // random state from starting distribution <br>
    2. &nbsp; $t \leftarrow 0$
    3. While S is not terminal:
        1. &nbsp; $A \sim \pi(\cdot\|S,\theta)$ <br>
        2. &nbsp; $S', R \leftarrow \text{env.step}(A)$ <br>
        3. &nbsp; $\delta \leftarrow R + \gamma \hat{V}(S',w) - \hat{V}(S,w)$ <br>
        4. &nbsp; $\textbf{w} = \textbf{w} + \alpha_\textbf{w} \delta \nabla_\textbf{w} \hat{V}(S,\textbf{w})$ &nbsp; // update critic <br>
        5. &nbsp; $\theta = \theta + \alpha_\theta \gamma^t \delta \nabla_\theta \log \pi(A\|S,\theta)$ &nbsp; // update actor <br>
        6. &nbsp; $t \leftarrow t + 1$ <br>
        7. &nbsp; $S \leftarrow S'$ <br>

__Output:__ parameters for actor: $\theta$, and critic: $\textbf{w}$
<hr>
- this implementation uses $\delta$ as an Advantage estimate (high variance)
- $\delta$ can be replaced by one of the variations discussed in the sections above
- pseudocode modified from Sutton&Barto [[6]][sab], Chapter 13
- great [Stackexchange post][why-gamma] for why we are using decay in the update of the actors parameters $\theta$.

### Todo

- the discounting problem (+ paper from discord)

- clean implementation


### Corresponding neuroanatomic structures for the Actor Critic mechanism

The functions of the two parts of the stratium (dorsal stratium -> action selection, ventral stratium -> reward processing) suggest that an Actor Critic mechanism is used for learning in our brains, where both the actor and the critic learn from the TD-Error $\delta$, which is produced by the critic. A TD-Error $\delta > 0$ would mean that the selected action led to a state with a better than expected value and if $\delta < 0$, it led to a state with a worse than average value. An important insight from Neuroscience is that the TD-Error corresponds to a pattern of dopamine neuron activations in the brain, rather than being just a scalar signal (in our brain, you could look at it as a vector of dopamine-neuron activity). These dopamine neurons modulate the updates of synapses in the actor and critic structures.

$$
\text{TD-Error} \; \delta \; \hat{=} \; \text{Activation pattern of dopamine neurons}
$$

The following image shows the corresponding structures in mammalian brains and how they interact.

<div class="img-block" style="width: 500px;">
    <img src="/images/actor-critic/reinforcement_learning_model_free_active_actor_critic_neural_implementation.png"/>
</div>
<center>Illustration from Massimiliano Patacchiola's blog [9]</center>

Experiments show that when the dopamine signal from the critic is distorted, e.g. by the use of cocaine, the subject was not able to learn the task (because the dopamine/error signal for the actor is too noisy).


### Final remark: Clean formalism

Reinforcement learning notation sometimes gets really messy and unpleasent to look at, to the point where it can be hard to absorb the important pieces of information. For this reason i think it is usually better to _omit some formalism and instead write clean looking formulas_ for the sake of readability, if the context of writing allows it (i.e. you are not writing a scientific paper). A piece that you can usually leave out if it is clear what we are referring to is $\theta$ in the subscript.


<!-- working gist: <script src="https://gist.github.com/till2/ace2a6cfd60c52994afa9536c412f8e5.js"></script> -->

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
1. Illustration of the Neural Net architecture with a shared body taken from [here][datahubbs-pic-link].
2. Pseudocode Image taken from [here][code].
3. [Sutton & Barto: Reinforcement Learning, An introduction (second edition)][sab]
4. [Hado van Hasselt: Lecture 8 - Policy Gradient][hadovanhasselt]
5. [HHU-Lecture slides:][semi-gradient] Approximate solution methods (for the semi-gradient definition)
6. [Stackexchange post][why-gamma]: Why we are using $\gamma$ as discounting to update the actors parameters $\theta$


### Pointers to other ressources
1. [Chris Yoon: Understanding Actor Critic Methods and A2C](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f)
2. [Richard Sutton: Actor-Critic Methods](http://incompleteideas.net/book/ebook/node66.html)
3. [Berkeley lecture slides](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf)
4. [CS885 Lecture 7b: Actor Critic](https://www.youtube.com/watch?v=5Ke-d1Itk3k)
5. [Actor Critic blogpost with illustrations and eligibility traces][actor-critic-blogpost]
6. [TD(0) Actor Critic code][actor-critic-TD0-code]
7. PyTorch Actor Critic [implementation][torch-actor-critic-code].
8. Nice ressource on A2C (1-step and n-step) with code [here][datahubbs-a2c].
9. [Massimiliano Patacchiola: ][awesome-well-written-rl-blog-series]

<!-- Ressources -->
[datahubbs-pic-link]: https://www.datahubbs.com/two-headed-a2c-network-in-pytorch/
[datahubbs-a2c]: https://www.datahubbs.com/policy-gradients-and-advantage-actor-critic/
[code]: https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
[torch-actor-critic-code]: https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
[actor-critic-TD0-code]: https://github.com/chengxi600/RLStuff/blob/master/Actor-Critic/Actor-Critic_TD_0.ipynb
[actor-critic-blogpost]: https://medium.com/geekculture/actor-critic-value-function-approximations-b8c118dbf723
[sab]: http://incompleteideas.net/book/the-book-2nd.html
[hadovanhasselt]: https://hadovanhasselt.files.wordpress.com/2016/01/pg1.pdf
[semi-gradient]: https://www.cs.hhu.de/fileadmin/redaktion/Fakultaeten/Mathematisch-Naturwissenschaftliche_Fakultaet/Informatik/Dialog_Systems_and_Machine_Learning/Lectures_RL/L4.pdf
[why-gamma]: https://ai.stackexchange.com/questions/10531/in-online-one-step-actor-critic-why-does-the-weights-update-become-less-signifi
[awesome-well-written-rl-blog-series]: https://mpatacchiola.github.io/blog/2017/02/11/dissecting-reinforcement-learning-4.html

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