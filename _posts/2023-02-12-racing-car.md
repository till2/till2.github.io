---
layout: post
title:  "Racing Car"
author: "Till Zemann"
date:   2023-02-12 20:31:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
positive_reward: true
reward: 2
tags: [reinforcement learning, machine learning, bachelor thesis]
thumbnail: "/images/racing-car/ferrari-inspired.jpeg"
---


<div class="img-block" style="width: 500px;">
    <img src="/images/racing-car/ferrari-inspired.jpeg"/>
</div>


<!-- <em style="float:right">First draft: 2022-10-22</em><br> -->

<!--
### Contents
* TOC
{:toc}
-->

### Introduction



### State of the art

- [Deep Reinforcement Learning for Autonomous Driving: A Survey](https://arxiv.org/abs/2002.00444) (really useful) (2021)
- [A Comprehensive Survey on the Application of Deep and Reinforcement Learning Approaches in Autonomous Driving](https://reader.elsevier.com/reader/sd/pii/S1319157822000970?token=2ECFFB83B4E92A712CA43828B1CA10E7689060C21AE7A39C0B9FC311025524456B85161D53D3B6ECEB2A3E47D0A48B7F&originRegion=eu-west-1&originCreation=20230212223824) (2021)
- [CNN+LSTM](https://ieeexplore.ieee.org/document/8500703)
- [David Ha, Jürgen Schmidhuber: Recurrent World Models Facilitate Policy Evolution](https://arxiv.org/pdf/1809.01999.pdf) (2018)
- [Antonin Raffin et. al. DECOUPLING FEATURE EXTRACTION FROM POLICY LEARNING: ASSESSING BENEFITS OF STATE REPRESENTATION LEARNING IN GOAL BASED ROBOTICS](https://arxiv.org/pdf/1901.08651.pdf) (2019)
- [Autonomous Vehicles on the Edge: A Survey on Autonomous Vehicle Racing](https://arxiv.org/pdf/2202.07008.pdf#cite.schwarting2021) (2022)
-> not much model-based RL research so far in autonomous driving, although i think it's super promising


### Related work: Model-based RL for autonomous driving


- [Deep Latent Competition: Learning to Race Using Visual Control Policies in Latent Space](https://arxiv.org/pdf/2102.09812.pdf) (2021)
- [Model-based versus Model-free Deep Reinforcement Learning for Autonomous Racing Cars](https://arxiv.org/pdf/2103.04909v1.pdf) (2021)


### Baselines

Model-free RL:
- LSTM-PPO
- PPO
- SAC


### My Approach

- Use [DreamerV3 - Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104v1) (2023)

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
1. Thumbnail taken from [Carblog: Autonomous Cars in Racing][thumb-website].

<!-- Ressources -->
[thumb-website]: https://www.carblog.co.uk/autonomous-cars-in-racing/


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
