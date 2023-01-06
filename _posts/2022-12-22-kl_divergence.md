---
layout: post
title:  "KL-Divergence"
author: "Till Zemann"
date:   2022-12-22 06:31:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
positive_reward: true
reward: 2
tags: [reinforcement learning, 🟢 not finished]
thumbnail: "/images/kl_divergence/2018-12-27-learning.png"
---


<div class="img-block" style="width: 500px;">
    <img src="/images/kl_divergence/2018-12-27-learning.png"/>
</div>


<!-- <em style="float:right">First draft: 2022-10-22</em><br> -->

<!--
### Contents
* TOC
{:toc}
-->

### Introduction

- divergence not distance, 
- not a distance because KL(a,b) != KL(b,a)
- (but usually used as distance -- ok if we use the correct order)

### Formula

$$
KL(\pi_1 || \pi_2) = \sum_{s \in S} \sum_{a \in A} \pi_1(a|s) \log \frac{\pi_1(a|s)}{\pi_2(a|s)}
$$



### Approximating KL Trick

- was in some blog, let me see...
- found it! URL: http://joschu.net/blog/kl-approx.html


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
1. Thumbnail taken from [Jessica Stringham's KL Divergence blogpost][jessica-stringham-kl-blogpost].


<!-- Ressources -->
[jessica-stringham-kl-blogpost]: https://jessicastringham.net/2018/12/27/KL-Divergence/


<!-- Optional Comment Section-->
{% if page.comments %}
<p class="vspace"></p>
<a class="commentlink" role="button" href="/comments/">Share your thoughts</a> <!-- role="button"  -->
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
