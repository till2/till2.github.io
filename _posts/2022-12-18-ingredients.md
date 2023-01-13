---
layout: post
title:  "Ingredients of Intelligence"
author: "Till Zemann"
date:   2022-12-18 14:25:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
positive_reward: true
reward: 2
tags: [reinforcement learning]
thumbnail: "/images/pillars/intelligence-coverart.png"
---

<!-- add the actor-critic diagram from Prof. Sutton.! -->

<div class="img-block" style="width: 300px;">
    <img src="/images/pillars/intelligence-coverart.png"/>
</div>

<em style="float:right">First draft: 2022-12-18</em><br>

<!--
### Contents
* TOC
{:toc}
-->

### Introduction

I recognize that the title sounds catchy, but being catchy is not the intention of this post. This is supposed to be a serious list of directions for future research (you can also just treat it as an overview).

This post includes what I believe is a somewhat comprehensive recipe for building intelligent systems, but surely some important ingredients are missing. Of course this list will be updated when I notice that an important concept is not included. 


### Definition

Before we get to the list, let's start by defining a measure for intelligence.

For that, i think the definition from [Shane Legg and Marcus Hutter][legg-hutter-intelligence] is nice (appropriate), because it makes general capability central:

$$
\begin{align*}
\Upsilon(\pi)   &\dot{=} \sum_{\mu \in E} 2^{-K(\mu)} V_\mu^\pi \\
                &\dot{=} \sum_{\mu \in E} 2^{-K(\mu)} \frac{1}{\Gamma} \mathbb{E}[ \sum_{i=1}^{\infty} \gamma^i r_i ] \\
                &\dot{=} \sum_{\mu \in E} 2^{-K(\mu)} \frac{1}{\sum_{i=1}^{\infty} \gamma^i} \mathbb{E}[ \sum_{i=1}^{\infty} \gamma^i r_i ]
\end{align*}
$$

where $\Upsilon(\pi)$ measures the universal intelligence of an agent with policy $\pi$. This universal intelligence is determined by the added performance (=value of the starting state in an environment) of different environments $\mu \in E$, with a weighting factor $2^{-K(\mu)}$ that weights the performance in simpler environments (=low [Kolmogorov complexity][wiki-kolmogorov-complexity]) higher.


### The list

<div class="table-wrap">
    <table class="table">
        <tr>
            <td><strong>Ingredient</strong></td>
            <td><strong>Purpose</strong></td>
            <td><strong>Implementation</strong></td>
        </tr>
        <tr>
          <td>Learning <br><em>(the most important one!)</em></td>
          <td>Aquiring new knowledge by updating your beliefs when your experience deviates from your expectation.</td>
          <td>Reinforcement Learning, Unsupervised Learning, Supervised Learning</td>
        </tr>
        <tr>
          <td>Curiosity</td>
          <td>Efficient Exploration.</td>
          <td>e.g. Feature-Space Curiosity</td>
        </tr>
        <tr>
          <td>Dreaming</td>
          <td>Recalling past experiences for consolidation into long-term memory and quicker learning.</td>
          <td>(Prioritized) Experience Replay</td>
        </tr>
        <tr>
          <td>World models & planning</td>
          <td>World models enable experiences in an imaginary world and if the world model is good, we can enable sample efficient learning because we don't need to interact with the real environment as much anymore. Planning means thinking ahead how trajectories will play out and using this information to select better actions. Planning is also only possible if we have a model of the world, so that we can look to see what might happen.</td>
          <td>Model-based RL</td>
        </tr>
        <tr>
          <td>Function approximation</td>
          <td>Compressing knowledge to generalize concepts (also as a sideeffect converting different modalities into thoughts and back into possibly other modalities).</td>
          <td>(Deep) Neural Networks</td>
        </tr>
        <tr>
          <td>Attention</td>
          <td>Focussing on some parts of the input data more than on other parts to make better predictions.</td>
          <td>Transformers</td>
        </tr>
  </table>
</div>

### TODO

- add pointers to research papers for each ingredient

- Curiosity: https://pathak22.github.io/noreward-rl/
- DreamerV3 (Model-based): https://arxiv.org/abs/2301.04104v1


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

1. Shane Legg and Marcus Huttter - [Universal Intelligence: A Definition of Machine Intelligence][legg-hutter-intelligence]
2. [Wiki: Kolmogorov complexity][wiki-kolmogorov-complexity]


<!-- Ressources -->
[legg-hutter-intelligence]: https://arxiv.org/pdf/0712.3329.pdf
[wiki-kolmogorov-complexity]: https://en.wikipedia.org/wiki/Kolmogorov_complexity



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