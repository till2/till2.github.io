---
layout: post
title:  "Actor Critics"
author: "Till Zemann"
date:   2022-12-16 00:32:41 +0200
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
<em style="float:right">Second draft: 2022-12-12</em><br>
<em style="float:right">Third draft: 2022-12-15</em>

<!--
### Contents
* TOC
{:toc}
-->

### Introduction

The Actor Critic is a powerful and beautiful method of learning, with surprising similarities to our dopaminergic learning circuits.

Let's start by looking at the REINFORCE algorithm, a method for training reinforcement learning (RL) agents. It is a policy gradient method, which means that it uses gradient ascent to adjust the parameters of the policy in order to maximize the expected reward. It does this by computing the gradient of the performance (goal) $J(\theta) \stackrel{.}{=} V^{\pi_\theta}(s_0)$ with respect to the policy parameters, and then updating the policy in the direction of this gradient. This update rule is known as the policy gradient update rule, and it ensures that the policy is always moving in the direction that will increase the expected future reward (=return). Because we need the entire return $G_t$ for the update at timestep $t$, REINFORCE is a Monte-Carlo method and theirfore only well-defined for episodic cases. 

One drawback of the pure REINFORCE algorithm is that it has a really high variance and could be unstable as a result. To lower the variance, we can substract a baseline $b(S_t)$, which has to be independent of the action. A good idea is to use state-values as a baseline, which reduce the magnitude of the expected reward (it has the effect of "shrinking" the estimated rewards towards the baseline value). Reducing the magnitude of the estimated rewards can help to reduce the variance of the algorithm. This is because the updates that the algorithm makes to the policy are based on the estimated rewards. If the magnitude of the rewards is large, the updates will also be large, which can cause the learning process to be unstable and can result in high variance. By reducing the magnitude of the rewards, the updates are also reduced, which can help to reduce the variance and thus stabilize the learning process.

The Actor-Critic algorithm is an extension of the REINFORCE algorithm that uses a value function as a baseline to improve the stability of the learning process. This baseline also needs to be learned (we have to approximate $V(s)$, usually using a _Deep Neural Network_), theirfore Actor-Critics are a combination of value-based and policy-based methods:


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
As a sidenote, $a(x) \propto b(x)$ states that a is proportional to b, meaning $a(x) = c \cdot b(x)$. This notation is sometimes useful for talking about gradients, because the factor $c$ is absorbed in the learning rate anyway.

We can extend the formula further by multiplying and deviding by <strong style="color: #ED412D">$\pi(a|S_t, \theta)$</strong> to get the expression <strong style="color: #ED412D">$\frac{\nabla \pi(a|S_t, \theta)}{\pi(a|S_t, \theta)}$</strong>. This is a common trick using the logarithm, where you can rewrite the gradient of $\log x$ with $\nabla \log x = \frac{1}{x} \nabla x = \frac{\nabla x}{x}$ (just using the chain rule). In our specific case, we can use this as  <strong style="color: #1E72E7">$\nabla \log \pi(a|S_t, \theta)$</strong> $=$ <strong style="color: #ED412D">$\frac{\nabla \pi(a|S_t, \theta)}{\pi(a|S_t, \theta)}$</strong>. 

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


### Actor and Critic as Deep Neural Networks

The main idea is that we update the actor parameters in the direction of a value that is estimated by the critic, e.g. the advantage. This makes sense because the critic is better able to evaluate the actual value of a state.

<div class="img-block" style="width:350px;float:right;margin-left:20px">
    <img src="https://www.datahubbs.com/wp-content/uploads/2018/08/two_headed_network.png"/>
</div>

As already mentioned, the actor is responsible for learning a policy $\pi(a\|s)$, which is a function that determines the next action to take in a given state. The critic, on the other hand, is responsible for learning a value function $V(s)$ or $Q(s,a)$, which estimates the future rewards that can be obtained by following the policy. The actor and critic work together to improve the policy and value function over time, with the goal of maximizing the overall rewards obtained by the system.

<em>Note, that it is common to use a shared neural network body. This is practical for learning features only once and not individually for both networks. The last layer of the body network connected to both the `policy head` and the `value head`), producing the outputs for actor and critic, respectively.</em>


### Actor Critic Algorithm

The following algorithm for an Actor Critic in the episodic case, we are calculating the TD-Error as $\delta \leftarrow R + \gamma \hat{V}(S',w) - \hat{V}(S,w)$, using our parameterized state-value function (the critic). This means that all bootstrappinging of the TD-Error depends on our current set of parameters, which can introduce a bias. Theirfore, the updates only include a part of the true gradient. These methods are called `semi-gradient` methods.


__Actor Critic algorithm (episodic):__

<hr>

__Input:__ 
policy parameterization $\pi(a|s,\theta)$  <em>(e.g. a Deep Neural Network)</em>,<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
state-value function parameterization $\hat{V}(s,\textbf{w})$ <em>(e.g. a Deep Neural Network)</em>,<br>

__Parameters:__ learning rates for the actor: $\alpha_\theta$, and for the critic: $\alpha_\textbf{w}$ <br>
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



### Variations 

1) If we want to get less variance and thus more stable updates, we could also calculate the advantage as the return $G_t \stackrel{.}{=} R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=t+1}^{\infty} \gamma^k R_{t+k+1}$ minus the state-value. For this variation of the actor-critic algorithm, we can only do updates after each episode (because we need to calculate the return $G_t$).

$$
\begin{align*}
A(s,a)  &= Q(s,a) - V(s) \\
            &= r + \gamma V(s') - V(s)
\end{align*}
$$

2) You could also estimate the advantage by using a critic Neural Network that estimates $\hat{V}(s)$ and $\hat{Q}(s,a)$ at the same time, and you just use $A(s,a) = Q(s,a) - V(s)$.

Overview of the Actor-Critic variations:

<div class="img-block" style="width: 500px;">
    <img src="/images/actor-critic/policy-gradient-variationen.png"/>
</div>


### Variance vs. Bias tradeoff for semi-gradient methods (using Function Approximation)

When estimating $Q(s,a)$ with semi-gradient methods, we have to trade of bias and variance. We can either use our observed rollout to calculate $Q(s,a)$, or use our state-value-function $V(s)$ (=our critic Neural Network):

$$
\begin{align*}
\hat{Q}(s_t,a_t)  &= \mathbb{E}[ r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots ]          \;\; (\text{ entire rollout}) \\
            &= \mathbb{E}[ r_{t+1} + \gamma V(s_{t+1}) ]                            \;\; (\text{ short rollout}) \\
            &= \mathbb{E}[ r_{t+1} + \gamma r_{t+2} + \dots + \gamma^k V(s_{t+k}) ] \;\; (\text{ k-step rollout}) \\
\end{align*}
$$

__For shorter k-step rollouts:__
- lower variance because the estimated state-value $\hat{V}(s)$ is an estimate based on lots of experience (=estimated average return).
- higher bias, because the estimate is produced by the Neural Net

__For longer k-step rollouts:__
- higher variance, because the return is only one observed monte-carlo rollout
- low bias, because we use the actual return (no bias if we use the complete rollout)

### Generalized Advantage Estimation (GAE)

Generalized Advantage Estimation (GAE) is a method for estimating the advantages. It is an extension of the standard advantage function, which is calculated as the difference between the expected return of an action and the expected return of the current state. GAE uses a discount factor (lambda) to weight the different k-step rollouts for estimating Q.

By weighting the different k-step rollouts, we try to optimize the variance vs. bias tradeoff without having to search for a good hyperparameter k (the length of the rollouts). Again, we assume that the influence of an action decreases exponentially with a parameter $\lambda \in [0,1]$ over time (be careful with this assumption, depending on the environment!).
The idea is pretty much the exact same as in TD($\lambda$).

- see my [TD($\lambda$) post][td-lambda-post] for more details.

$$
\begin{align*}
\hat{Q}_t^\text{GAE}  &= (1 - \lambda) (\hat{Q}_t^{(k=1)} + \lambda \hat{Q}_{t}^{(k=2)} + \lambda^2 \hat{Q}_{t}^{(k=3)} + \dots) \\
                      &= (1 - \lambda) \sum_{k=0}^{T} \lambda^k \hat{Q}_{t}^{(k)}
\end{align*}
$$


### Implementation

First, we need to rewrite the updates in terms of losses $\mathcal{L}_{\text{critic/actor}}$ that can be backpropagated in order to use modern Deep Learning libraries. 


For the critic, we just want to approximate the true state-values. To do that, we move them closer to the observed return $G_t$, using either a squared error or the [Huber-loss][torch-huber-loss], that doesn't put as much weight on outliers. We can also use some $L_2$ regularization on the Delta of parameters $\Delta \textbf{w} = \textbf{w}_{\text{new}} - \textbf{w}$ to ensure that we don't update our parameters too much (this is always a good idea in Reinforcement Leraning, because bad policy leads to bad data in the future):

$$
\begin{align*}
\mathcal{L}_{\text{critic}} &= \| \hat{Q}(s_t,a_t) - \hat{V}(S_t,\textbf{w}) \|_2^2 + \kappa \| \textbf{w}_{\text{new}} - \textbf{w} \|_2^2
\end{align*}
$$

<em>I didn't add the regularization to the implementation yet, because i couldn't figure out how to implement it in PyTorch so far (you would have to backpropagate the loss first and then calculate the squared $L_2$ norm of the new and old parameters), so treat it as if we set $\kappa=0$ :-( </em>.

For the actor, we are using the advantage $A(s, a)$ instead of $\delta$ now, which is one of the shown variations. To get our loss, we can just pull in the factor $A(S,A,\textbf{w})$ like this: $a \nabla_x f(x) = \nabla_x a f(x)$ (the factor $A(S,A,\textbf{w})$ doesn't depend on $\theta$, otherwise this wouldn't be valid). 

Also note that $\gamma^t$ is already baked into the discounted return $G_t$. Discounting makes a lot of sense for many environments, because the action probably has a higher effect short-term and doesn't matter that much long-term. You should consider your environment carefully here though, because in some cases, for example the game Go, action might have a long-term influence/effect. 
[[6]][pieter-abbeel-discounting]

For readability, i'll leave out the $t$ in the subscript in the following formulas, but all $S$, $A$ and $Q$ are in timestep $t$. Lastly notice that we'll have to negate the entire term to get a loss function, because now instead of maximizing the term, we minimize the negative term (which is the same).

$$
\begin{align*}
\theta                                  &= \theta + \alpha_\theta A(S,A,\textbf{w}) \nabla_\theta \log \pi(A|S,\theta) \\
                                        &= \theta + \alpha_\theta \nabla_\theta A(S,A,\textbf{w}) \log \pi(A|S,\theta) \\ \\
\Rightarrow \mathcal{L}_{\text{actor}}  &= - A(S,A,\textbf{w}) \log \pi(A|S,\theta) \\
                                        &= - [\hat{Q}(s,a) - V(S,\textbf{w})] \log \pi(A|S,\theta)
\end{align*}
$$

<em> Note that $\hat{Q}(s,a)$ is estimated with a k-step rollout (see section: "Variance vs. Bias tradeoff"). </em>
Another choice that we'll have to make after letting the episode play out is whether we want to update the networks for each timestep or only once. If you chose the latter, you can just sum over all the individual losses to get one loss for the episode:

$$
\begin{align*}
\mathcal{L}_{\text{actor}}  &= - \sum_{t=1}^{T} A(S_t,A_t,\textbf{w}) \log \pi(A_t|S_t,\theta) \\
                            &= - \textbf{advantages}^\top \textbf{logprobs}

\end{align*}
$$




### Corresponding neuroanatomic structures to Actor and Critic

The functions of the two parts of the stratium (dorsal stratium -> action selection, ventral stratium -> reward processing) suggest that an Actor Critic mechanism is used for learning in our brains, where both the actor and the critic learn from the TD-Error $\delta$, which is produced by the critic. A TD-Error $\delta > 0$ would mean that the selected action led to a state with a better than expected value and if $\delta < 0$, it led to a state with a worse than average value. An important insight from Neuroscience is that the TD-Error corresponds to a pattern of dopamine neuron activations in the brain, rather than being just a scalar signal (in our brain, you could look at it as a vector of dopamine-neuron activity). These dopamine neurons modulate the updates of synapses in the actor and critic structures.

$$
\begin{align*}
\text{TD-Error} \; \delta \; &\hat{=} \; \text{Activation pattern of dopamine neurons} \\
                             &\hat{=} \; \text{experience - expectated experience}
\end{align*}
$$

The following image shows the corresponding structures in mammalian brains and how they interact.

<div class="img-block" style="width: 500px;">
    <img src="/images/actor-critic/reinforcement_learning_model_free_active_actor_critic_neural_implementation.png"/>
</div>
<center>Illustration from Massimiliano Patacchiola's blog [9]</center>

Experiments show that when the dopamine signal from the critic is distorted, e.g. by the use of cocaine, the subject was not able to learn the task (because the dopamine/error signal for the actor is too noisy).


### Final remark: Clean formalism

Reinforcement learning notation sometimes gets really messy and unpleasent to look at, to the point where it can be hard to absorb the important pieces of information. For this reason i think it is usually better to _omit some formalism and instead write clean looking formulas_ for the sake of readability, if the context of writing allows it (i.e. you are not writing a scientific paper). A piece that you can usually leave out if it is clear what we are referring to is $\theta$ in the subscript.


### TODO

- the discounting problem (+ paper from discord)
- clean implementation

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
2. [Stackexchange post][why-gamma]: Why we are using $\gamma$ as discounting to update the actors parameters $\theta$
3. [Sutton & Barto: Reinforcement Learning, An introduction (second edition)][sab]
4. [Hado van Hasselt: Lecture 8 - Policy Gradient][hadovanhasselt]
5. [HHU-Lecture slides:][semi-gradient] Approximate solution methods (for the semi-gradient definition)
6. [Pieter Abbeel: L3 Policy Gradients and Advantage Estimation (Foundations of Deep RL Series)][pieter-abbeel-discounting]
7. [Daniel Takeshi's blog: Notes on the Generalized Advantage Estimation Paper][gae-danieltakeshi-blog]
8. [PyTorch docs: HuberLoss][torch-huber-loss]


### Pointers to other ressources
1. [Chris Yoon: Understanding Actor Critic Methods and A2C](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f)
2. [Richard Sutton: Actor-Critic Methods](http://incompleteideas.net/book/ebook/node66.html)
3. [Berkeley lecture slides](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf)
4. [CS885 Lecture 7b: Actor Critic](https://www.youtube.com/watch?v=5Ke-d1Itk3k)
5. [Actor Critic blogpost with illustrations and eligibility traces][actor-critic-blogpost]
6. [TD(0) Actor Critic code][actor-critic-TD0-code]
7. PyTorch Actor Critic [implementation][torch-actor-critic-code].
8. Nice ressource on A2C (1-step and n-step) with code [here][datahubbs-a2c].
9. [Massimiliano Patacchiola: Dissecting Reinforcement Learning-Part.4: Actor-Critic (AC) methods][awesome-well-written-rl-blog-series] (+ Correlations to Neuroanatomy)

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
[pieter-abbeel-discounting]: https://youtu.be/AKbX1Zvo7r8?t=1872
[td-lambda-post]: /blog/2022/12/07/td_lambda
[gae-danieltakeshi-blog]: https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/
[torch-huber-loss]: https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html

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