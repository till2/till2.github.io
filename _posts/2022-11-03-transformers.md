---
layout: post
title:  "Transformers (👷)"
author: "Till Zemann"
date:   2022-11-02 20:36:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
---

<!--
### Contents
* TOC
{:toc}
-->

<!-- builder image -->
<div class="img-block" style="width: 450px;">
    <img src="/images/builder_one.png"/>
</div>
<center>Site under construction. 👷</center>

### Introduction

Transformers are one of the coolest inventions of recent years in Natural Language Processing (NLP) and Machine Learning in general. This technology is a combination of neural networks, basic database concepts and attention with energy-functions and they parallelize amazingly well (one part that contributes to this is the "multi-headed attention 🐲🐲", but more on that later).


### Attention

### Positional encoding

### Architecture

<div class="img-block" style="width: 1000px;">
    <img src="/images/transformers/transformer.png"/>
</div>

### Todo
- add lecture slide notes/ handwritten drawings
- add notes from: [self-attention video](https://youtu.be/yGTUuEx3GkA)
- watch and write down notes for:
1. [11-785 Deep Learning Recitation 11: Transformers Part 1](https://www.youtube.com/watch?v=X2nUH6fXfbc) <- implementation
2. [Attention (Aleksa Gordić)](https://www.youtube.com/watch?v=n9sLZPLOxG8) <- this is part 1
3. [Transformer (Aleksa Gordić)](https://www.youtube.com/watch?v=cbYxHkgkSVs) <- and part 2

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


### References
1. [Transformer architecture image][transformer-img].

<!-- Ressources -->
[transformer-img]: https://deepfrench.gitlab.io/deep-learning-project/resources/transformer.png


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