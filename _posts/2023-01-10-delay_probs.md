---
layout: post
title:  "Multi-Agent Path Finding with Delay Probs"
author: "Till Zemann"
date:   2023-01-08 14:31:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
positive_reward: true
reward: 2
tags: [uni, presentation]
thumbnail: "/images/delay_probs/thumbnail.png"
---


<div class="img-block" style="width: 500px;">
    <img src="/images/delay_probs/thumbnail.png"/>
</div>

<!--
<em style="float:right">First draft: 2023-01-07</em><br>
-->

<!--
### Contents
* TOC
{:toc}
-->


### Introduction

ABC


### Todo

- read the paper


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
1. Thumbnail taken from [here][mapf-w-delay-probs-paper].
2. [Hang Ma and T. K. Satish Kumar and Sven Koenig. "Multi-Agent Path Finding with Delay Probabilities" AAAI (2017).][mapf-w-delay-probs-paper]


<!-- Ressources -->
[mapf-w-delay-probs-paper]: https://arxiv.org/abs/1612.05309

<!-- Optional Comment Section-->
{% if page.comments %}
<p class="vspace"></p>
<a class="commentlink" role="button" href="/comments/">Share your thoughts.</a> <!-- role="button"  -->
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
