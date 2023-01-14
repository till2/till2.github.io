---
layout: post
title:  "Multi-Agent Path Finding with Delay Probs"
author: "Till Zemann"
date:   2023-01-10 14:31:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
positive_reward: true
reward: 2
tags: [uni, presentation]
thumbnail: "/images/delay_probs/fig1.png"
---

<!-- thumbnail.png -->


<div class="img-block" style="width: 500px;">
    <img src="/images/delay_probs/fig1.png"/>
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



### Adressed Problem

- current state of the art: solvers can solve large normal MAPF-instances
- replanning often required during runtime due to delayed agents 


### MAPF-DP formalized

- most is same as normal MAPF (undirected graph, discrete time steps)

New:
- agents have delay probabilites when executing a move-action (they stay in the current vertex with the `delay prob` and execute the move-action successfully with a probability of `1 - delay prob`)
- wait actions are always executed successfully
->  utilizing this probabilistic information reduces the need for frequent (time-instensive) replanning and the number of failures in plan execution

Problem formalization:
- find 1) a _MAPF-DP_ plan (non-conflicting paths for all agents) without conflicts _and_ 2) a _plan-execution policy_ that executes the plan so that no collisions occur during runtime

Prohibited Conflicts:
- vertex-conflict
- edge-conflict
- follow-conflict

<!-- ### Related work / MDP variants -->

### Notation





### Plan-Execution Policy



### Robust Plan-Execution Policies


#### Fully Synchronized Policies

- robust

#### Minimal Communication Policies

- robust
- partial order (?)
- agent $a_j$ sends message to all other agents when it reaches a new local state
- _transitive reduction_ (?) minimizes the number of edges in the directed graph -> this means it minimizes the number of messages between agents 



### Example

<p class="vspace"></p>
Example MAPF-DP instance:
<div class="img-block" style="width: 500px;">
    <img src="/images/delay_probs/fig_1.png"/>
</div>
<p class="vspace"></p>

Partial order (fig2) and transitive reduction for minimal communication policies (fig3): 
<div class="img-block" style="width: 500px;">
    <img src="/images/delay_probs/fig_2_3.png"/>
</div>

- evtl. animiert auf die Folie (mit Schritten die agents bewegen und daneben die Graphen revealen) - CS50AI :)



### Approximate Minimization in Expectation (AME) MAPF-DP solver

- use the AME solver to generate valid MAPF-DP plans and feed the plans to Minimal Communication Policies (MCPs) to get results with small Makespans

- needs to approximate the average Makespan (for execution) so it can generate a good plan (with a low average Makespan) 


<!-- include graphs -->




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
1. [Hang Ma and T. K. Satish Kumar and Sven Koenig. "Multi-Agent Path Finding with Delay Probabilities" AAAI (2017).][mapf-w-delay-probs-paper]
2. Thumbnail taken from [1]


<!-- Ressources -->
[mapf-w-delay-probs-paper]: https://arxiv.org/abs/1612.05309

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
