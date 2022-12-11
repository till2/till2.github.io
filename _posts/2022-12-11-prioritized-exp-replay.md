---
layout: post
title:  "The Prioritized Experience Replay Buffer"
author: "Till Zemann"
date:   2022-12-08 02:00:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
positive_reward: true
reward: 1
tags: [reinforcement learning, memory, dataset]
thumbnail: "/images/experience-replay/experience-replay-buffer.png" 
---


<!-- for multiple tags use a list: [hello1, hello2] -->

<!--
### Contents
* TOC
{:toc}
-->

<!--
TODO:
- add image links to References
-->

### Intro

Prioritized Experience Replay (PER) is a technique that was introduced by [[5]][prioritized-experience-replay] Schaul et al. in 2015. It is a method to improve the performance of the Experience Replay (ER) buffer. The idea is to sample the experiences in the buffer with a probability that is proportional to the TD-error of the experience, because we can learn the most for transitions that have a high TD-error (where the agent was the wrong-est).

<div class="img-block" style="width: 900px;">
    <img src="/images/experience-replay/experience-replay-buffer.png"/>
</div>


### Discussion

I would argue that the dual-memory (short-term and long-term) of [[6]][catastrophic-forgetting-dual-memory] Atkinson et. al. is not necessary as

1. the short-term learning is achieved through learning from the (prioritized) replay buffer and
2. the long-term knowledge is distilled in the neural network.  

For that to work, we need:

1. that old (long-term) knowledge isn't forgotten by the network because it got pushed out by newer knowledge,
   theirfore we want to capture the entire data distribution while not slowing down learning by much.

We can achieve that by compressing old transitions into new (much fewer!) transitions that the agents learns from, or 

- we want to capture the entire distribution while favoring newer data, theirfore we need a mechanism that measures the similarity of new data compared to the old data and a bonus for freshness

- tradeoff between freshness of data and dis-similarity between that piece of data compared to the entire dataset (we can approximate the distribution of the dataset by the current entries of the replay buffer or by the knowledge compressed in the neural network (question: how could you do that?) ) (capturing edge-cases, might be more important to remember)

after the replay buffer is filled up, we need a measure how similar a new


How can you measure the similarity of data?
- first idea: learn an embedding function and compare the vectors, e.g. by the cosine-similarity function



<!-- In-Text Citing -->
<!-- 

Referencing equations:
$$
\begin{equation} \tag{1}\label{eq:1}
x=y
\end{equation}
$$
I reference equation \eqref{eq:1}


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

```c
for(int i=0; i<comm_sz; i++){
	print("%d\n", i);
}
```

<div class="output">
result: 42
</div>

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

1. Thumbnail & Zhang, Sutton's paper summarized at [endtoend.ai][endtoendai]: Paper Unraveled: A Deeper Look at Experience Replay (Zhang and Sutton, 2017).
2. [TheComputerScientist (YouTube)][thecomputerscientist]: How To Speed Up Training With Prioritized Experience Replay 
3. [endtoendai - one-slide-summary:][endtoendai-fundamentals-of-ER] Revisiting Fundamentals of Experience Replay
4. [Shangtong Zhang, Richard S. Sutton: A Deeper Look at Experience Replay][zhnang-sutton]
5. [Schaul, Quan, Antonoglou and Silver (2015): Prioritized Experience Replay][prioritized-experience-replay]
6. [Atkinson et. al.][catastrophic-forgetting-dual-memory] Pseudo-Rehearsal: Achieving Deep Reinforcement
Learning without Catastrophic Forgetting

<!-- Ressources -->
[RESSOURCE]: LINK
[endtoendai]: https://www.endtoend.ai/paper-unraveled/cer/
[endtoendai-fundamentals-of-ER]: https://www.endtoend.ai/one-slide-summary/revisiting-fundamentals-of-experience-replay/
[thecomputerscientist]: https://www.youtube.com/watch?v=MqZmwQoOXw4
[zhnang-sutton]: https://arxiv.org/abs/1712.01275
[prioritized-experience-replay]: https://arxiv.org/abs/1511.05952
[catastrophic-forgetting-dual-memory]: https://arxiv.org/pdf/1812.02464.pdf

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