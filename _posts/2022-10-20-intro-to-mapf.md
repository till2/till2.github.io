---
layout: post
title:  "Multi-Agent Pathfinding"
author: "Till Zemann"
date:   2022-10-20 15:57:00 +0200
categories: jekyll update
math: true
---

<!-- Execute in /_posts/: -->
<!-- pandoc -o first_paper_summary.pdf 2022-09-27-intro-to-mapf.md -->

* TOC
{:toc}

## The Multi-Agent Pathfinding Problem

Multi-Agent Pathfinding (__MAPF__) is the problem of planning paths for multiple agents without colliding.


## Assumptions

Common assumptions are:

- the environment is __discrete__
- an agent executes one action per timestep
- an agent occupies one vertex/node per timestep


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


## Conflicts

To properly define a MAPF problem, you should cover which of the following conflicts are allowed and which can not appear in a solution $\pi$.

__Conflict types__:

![](images/conflict-types.png "img1")  <!-- For pandoc (md to pdf) -->
![](/images/conflict-types.png "img2") <!-- For the website        -->


## Objectives and Constraints

The two most used objective functions are the __Makespan__ and __Sum of costs__.

#### Makespan

The __Makespan__ of a MAPF solution is definded as the number of timesteps it takes until all agents reach their goals:
<strong style="color: #d98404" >$J(\pi) = \max_{1 \leq i \leq k}|\pi_i|$</strong>

#### Sum of costs

The __sum of costs__ objective function takes the length of all individual agent plans into consideration by summing over all action plan lengths: <strong style="color: #d98404" >$J(\pi) = \sum_{1 \leq i \leq k}|\pi_i|$</strong>

There is also the not so common __sum of fuel__ objective function that counts all non-waiting moves.

An __optimal solution__ to our problem is one that __minimizes the chosen objective function__ <strong style="color: #d98404" >$J(\pi)$</strong> (and satisfies all other given constraints).

#### Constraints

Typical hard constraints that are additionally added are __k-robustness__ (an agent can only move to a vertex that hasn't been visited by any agent for $k$ timesteps) and __formation rules__.
The __k-robustness__ adresses the possiblity of delays that could result in agents colliding at execution. The goal is to be within a probabilistic margin for conflicts or have a policy that can deal with delays at execution time to prevent conflicts.
__Formation rules__ enforce a specific formation of agents, e.g. to allow communication chains via neighboring agents.

## Target behaviors

If an agent that already arrived at its target position doesn't plan on moving away from the target while waiting for other agents to reach their goals it is common to not count the waiting moves towards the sum of cost. 

There are two possibilities of handling agents that reach their target. The agent can either __stay at the target__ or __disappear at the target__. The stay at target behavior is more commonly used because it doesn't assume that the environment has a special mechanism for handling the transportation of the agent (e.g. to a fixed starting position) upon arriving at the target.


## Special MAPF problems

#### Weighted Actions

MAPF with weighted actions addresses problems, where the assumption of one action per timestep is not useful. The length of an action can be encoded as the weights in an weighted graph, which can be represented as $2^k$-grids or in a generalized form as euclidian (2d) space.
Note, that diagonal moves in euclidian space are possible and have an execution (time-) cost of $\sqrt{2}$.

#### Motion-planning

This takes the MAPF problem to a __state-based__ problem, where the state encodes information like position, orientation and velocity. AN edge between two state configurations can be seen as planning movement (or kinematic motion). If kinematic constraints are added to the MAPF problem, the graph becomes __directed__.

A (not comprehensive) list of other extensions of MAPF includes

- MAPF with large agents
- MAPF with kinematic constraints
- Anonymous MAPF
- Colored MAPF
- Online-MAPF.



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
1. Stern, R., Sturtevant, N., Felner, A., Koenig, S., Ma, H., Walker, T., ... & Boyarski, E. (2019). Multi-agent pathfinding: Definitions, variants, and benchmarks. In AAAI/ACM Conference on AI, Mobility, and Autonomous Systems (pp. 75-82). 
([arXiv][marl-defs-variants-benchmarks])
2. Kaduri, Omri: From A* to MARL ([5 part blogpost series][kaduri-mapf-to-marl])

<!-- Ressources -->
[marl-defs-variants-benchmarks]: https://arxiv.org/abs/1906.08291
[kaduri-mapf-to-marl]: https://omrikaduri.github.io/