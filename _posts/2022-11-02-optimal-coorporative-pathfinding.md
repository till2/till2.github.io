---
layout: post
title:  "Optimal Coorporative Pathfinding (wip)"
author: "Till Zemann"
date:   2022-11-02 20:36:41 +0200
categories: jekyll update
math: true
---
* TOC
{:toc}

## Structure

Intro
- What is this paper about?
- Why is it relevant?
- How is it achieved?
- What is the structure of this summary?

Sections

Conclusion
- discussion/ personal comment


## Todo

INTRO
- applications
- single-agent A*
- properties: optimal, fast, universal with modifications (in PROBLEM FORMULATION section)

RELATED WORK
- LRA* and it's drawbacks
- HCA* with reservation table (Silver)

PROBLEM FORMULATION
- grid representation
- allowed conflicts 

STANDARD ADMISSABLE ALGO (centralized)
- CSP Formulation

OPERATOR DECOMPOSITION (OD)
- reduce branching factor
- pre/post move positoins
- avoid expanding intermediate nodes -> prune nodes
- how to implement it
- properties: admissable and complete
- duplicate detection
- advanced duplicate detection

HEURISTIC
- admissable (no overestimates)
- SOC heuristic precomputed with BFS or RRA*

INDEPENDENT SUBPROBLEMS + INDEPENDENCE DETECTION ALGO (ID)
- explain ...

EXPERIMENTAL RANKINGS

CONCLUSION
- use OD + ID to make coorporative MAPF practical (big improvement over standard admissable algorithm)

## 

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


## References
1. [Kaggle - Transformers from Scratch][kaggle-transformer-from-scratch]

<!-- Ressources -->
[kaggle-transformer-from-scratch]: https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch/notebook