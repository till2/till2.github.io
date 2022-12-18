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

This post includes what I believe is a somewhat comprehensive recipe for building intelligent systems, but surely some important ingredients are missing. I'll update the list when I notice that an important concept is not included. 

This list can be treated as an overview and as directions for future research.

### The list

<div class="table-wrap">
    <table class="table">
        <tr>
            <td><strong>Ingredient</strong></td>
            <td><strong>Purpose</strong></td>
            <td><strong>Implementation</strong></td>
        </tr>
        <tr>
          <td>Learning</td>
          <td>Aquiring new knowledge by updating your beliefs when your experience deviates from your expectation.</td>
          <td>Reinforcement Learning, Unsupervised Learning, Supervised Learning</td>
        </tr>
        <tr>
          <td>Curiosity</td>
          <td>Efficient Exploration.</td>
          <td>Curiosity baked into the objective-function.</td>
        </tr>
        <tr>
          <td>Dreaming</td>
          <td>Recalling past experiences for consolidation into long-term memory and quicker learning.</td>
          <td>(Prioritized) Experience Replay</td>
        </tr>
        <tr>
          <td>Planning</td>
          <td>Thinking ahead how trajectories will play out and using this information to select better actions.</td>
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

<!-- 
### References
1. Illustration of the Neural Net architecture with a shared body taken from [here][datahubbs-pic-link].
2. [Stackexchange post][why-gamma]: Why we are using $\gamma$ as discounting to update the actors parameters $\theta$
3. [Sutton & Barto: Reinforcement Learning, An introduction (second edition)][sab]
4. [Hado van Hasselt: Lecture 8 - Policy Gradient][hadovanhasselt]
5. [HHU-Lecture slides:][semi-gradient] Approximate solution methods (for the semi-gradient definition)
-->


<!-- Ressources -->
<!--
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

-->

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