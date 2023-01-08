---
layout: post
title:  "Multishot Solving"
author: "Till Zemann"
date:   2023-01-08 14:31:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
positive_reward: true
reward: 2
tags: [uni, ASP]
thumbnail: "/images/multishot_solving/robinhoodmultishot_5559.webp"
---


<div class="img-block" style="width: 500px;">
    <img src="/images/multishot_solving/robinhoodmultishot_5559.webp"/>
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




### Iterative multishot solving

The idea is to only ground the base once (at the beginning) and then only ground the things that change with the horizon $t$ at each step to save some time on grounding. The base and step groundings are then combined to get the program for horizon $t$:

<svg width="800" height="200" version="1.1" xmlns="http://www.w3.org/2000/svg">
	<ellipse stroke="black" stroke-width="1" fill="none" cx="247.5" cy="57.5" rx="70" ry="30"/>
	<text x="192.5" y="63.5" font-family="Times New Roman" font-size="20">grounded base</text>
	<ellipse stroke="black" stroke-width="1" fill="none" cx="515.5" cy="57.5" rx="70" ry="30"/>
	<text x="450.5" y="63.5" font-family="Times New Roman" font-size="20">grounded step(t)</text>
	<ellipse stroke="black" stroke-width="1" fill="none" cx="377.5" cy="151.5" rx="145" ry="30"/>
	<text x="259.5" y="157.5" font-family="Times New Roman" font-size="20">combined grounded program</text>
	<ellipse stroke="black" stroke-width="1" fill="none" cx="377.5" cy="151.5" rx="150" ry="35"/>
	<polygon stroke="black" stroke-width="1" points="271.811,75.078 353.189,133.922"/>
	<polygon fill="black" stroke-width="1" points="353.189,133.922 349.636,125.182 343.777,133.286"/>
	<polygon stroke="black" stroke-width="1" points="490.706,74.389 402.294,134.611"/>
	<polygon fill="black" stroke-width="1" points="402.294,134.611 411.721,134.24 406.091,125.975"/>
</svg>


### Parallel multishot solving






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
1. Thumbnail taken from [here][thumbnail].
2. [Martin Gebser, Roland Kaminski, Benjamin Kaufmann and Torsten Schaub: Multi-shot ASP solving with Clingo][multishot-solving-paper]


<!-- Ressources -->
[thumbnail]: https://static.tvtropes.org/pmwiki/pub/images/robinhoodmultishot_5559.jpg
[multishot-solving-paper]: https://arxiv.org/pdf/1705.09811.pdf

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
