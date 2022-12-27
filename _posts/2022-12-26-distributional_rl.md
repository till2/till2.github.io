---
layout: post
title:  "Distributional Reinforcement Learning"
author: "Till Zemann"
date:   2022-12-26 20:31:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
positive_reward: true
reward: 2
tags: [reinforcement learning, stochastics, 🟢 not finished, research]
thumbnail: "/images/distributional_rl/Fig_00.png"
---


<div class="img-block" style="width: 500px;">
    <img src="/images/distributional_rl/Fig_00.png"/>
</div>


<!-- <em style="float:right">First draft: 2022-10-22</em><br> -->

<!--
### Contents
* TOC
{:toc}
-->

### Introduction

- introduction to distributional rl paper: [Marc G. Bellemare, Will Dabney, Rémi Munos - A distributional perspective on reinforcement learning (August 2017)][distributional-rl-paper]
- read the book: ["Distributional RL (2023), MIT Press" -- html variant currently][distributional-rl-book]
- [Will Dabney, Georg Ostrovski, David Silver, Rémi Munos: Implicit Quantile Networks for Distributional Reinforcement Learning (2018)][will-dabney-deepmind-paper]
- [WikiDocs: Distributional RL (overview) blogpost][wikidocs]


### Benefits

- safe Reinforcement Learning (?)


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
1. [Marc G. Bellemare, Will Dabney, Rémi Munos - A distributional perspective on reinforcement learning (August 2017)][distributional-rl-paper]
2. Marc G. Bellemare and Will Dabney and Mark Rowland: [Distributional Reinforcement Learning][distributional-rl-book]
3. [Will Dabney, Georg Ostrovski, David Silver, Rémi Munos: Implicit Quantile Networks for Distributional Reinforcement Learning (July 2018)][will-dabney-deepmind-paper]
4. Thumbnail from (and source): [WikiDocs: 34. Distributional RL][wikidocs].


<!-- Ressources -->
[thumbnail-paper]: https://arxiv.org/pdf/2007.04309.pdf
[distributional-rl-paper]: https://arxiv.org/pdf/1707.06887.pdf
[distributional-rl-book]: https://www.distributional-rl.org/
[will-dabney-deepmind-paper]: https://willdabney.com/publication/iqn/
[wikidocs]: https://wikidocs.net/175856

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
