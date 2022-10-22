---
layout: post
title:  "Multi-Agent Pathfinding"
author: "Till Zemann"
date:   2022-10-20 15:57:00 +0200
categories: jekyll update
math: true
---

* TOC
{:toc}

## The Multi-Agent Pathfinding Problem

Multi-Agent Pathfinding (__MAPF__) is the problem of planning paths for multiple agents without colliding.


## Assumptions

We assume that the environment is __discrete__ and __2-dimensional__, e.g. robots moving in a warehouse are modelled as points in a grid.
There probably are good algorithms for finding solutions in more complex scenarios, such as:

- 3d spaces
- continuous environments
- probabilistic environment dynamics

but we will stick to the easier problems in our MAPF course.


## Input

The input to a MAPF problem is a triple <strong style="color: #1E72E7">$<G,s,t>$</strong> consisting of:

- an undirected graph <strong style="color: #1E72E7">$G = (V,E)$</strong>

- a mapping <strong style="color: #1E72E7">$s$</strong> to source vertices with 
<strong style="color: #1E72E7">$s: [1,\dots,k] \to V$</strong>

- a mapping <strong style="color: #1E72E7">$t$</strong> to target vertices with 
<strong style="color: #1E72E7">$t: [1,\dots,k] \to V$</strong>


## Solution

The solution of a MAPF problem is a set <strong style="color: #039947">$\pi$</strong> of single-agent plans without conflicts: 
<strong style="color: #039947">$\pi$ = {$\pi_1, \pi_2, \dots, \pi_k$}</strong> where $\pi_i$ denotes the single-agent plan for agent $i$. 
A single-agent plan is an action mapping $\pi$ (careful: notation overload!) that results in the agent being their target state. We can write this constraint as <strong style="color: #039947">$\pi_i[|\pi|] = t(i)$</strong>.

Note, that $\pi$ does __not__ include the starting position $s(i)$.
Instead, the first entry in $\pi$ is the action that performed on the first timestep.

We can also ask, where an agent $i$ is positioned after timestep $x$ (equivalent to asking which node an agent occupies). We would write this as <strong style="color: #039947">$\pi_i[x]$</strong>.


## Conflict types

To properly define a MAPF problem, the definition should cover which of the following situations are considered to be conflicts and theirfore can not appear in a solution $\pi$:

![](images/conflict-types.png "img1")  <!-- For pandoc (md to pdf) -->
![](/images/conflict-types.png "img2") <!-- For the website        -->

## Commonly used Algorithms




<!-- In-Text Citing -->
<!-- 
You can...
- use bullet points
1. use
2. ordered
3. lists


do $X$ math

embed images:
<div class="img-block" style="width: 800px;">
    <img src="/images/lofi_art.png"/>
    <span><strong>Fig 1.1.</strong> Agent and Environment interactions</span>
</div>

refer to links:
[(k-fold) Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)

{% highlight python %}
@jit
def f(x)
    print("hi")
# does cool stuff
{% endhighlight %}
-->

## References
1. Kaduri, Omri: From A* to MARL ([5 part blogpost series][kaduri-mapf-to-marl])

<!-- Ressources -->
[kaduri-mapf-to-marl]: https://omrikaduri.github.io/