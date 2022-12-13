---
layout: post
title:  "The Prioritized Experience Replay Buffer"
author: "Till Zemann"
date:   2022-12-11 02:00:41 +0200
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

<em>This is a draft for a research idea.</em>

### Intro

Prioritized Experience Replay (PER) is a technique that was introduced by [[5]][prioritized-experience-replay] Schaul et al. in 2015. The method aims to improve the performance of the Experience Replay (ER) buffer that lets Reinforcement Learning agents learn quicker by replaying transitions that were observed by another (old) policy (=off-policy learning). The idea is to sample experiences in the buffer with a probability that is proportional to the TD-error of the experience, because we can learn the most from transitions that have a high TD-error (where the agent was the wrong-est).

Replay buffers resemble some similarity to humans learning in their sleep by replaying important series of transitions (=dreaming) that were experienced during the day to consolidate them into long-term memory [[7]][wiki-sleep-learning] and learn from them to develop problem-solving skills [[8]][wiki-dreams].

<div class="img-block" style="width: 900px;">
    <img src="/images/experience-replay/experience-replay-buffer.png"/>
</div>
<center>Illustration of the replay buffer (usually implemented as a deque) interface for off-policy learning.</center>

### Discussion

I would argue that complex architectures aimed at mitigating catastrophic forgetting, such as the dual-memory architecture (short-term and long-term) of [[6]][catastrophic-forgetting-dual-memory] Atkinson et. al. are not necessary if the following conditions are met:

1. the short-term learning is achieved through learning from the (prioritized) replay buffer (or could also be achieved by combined experience replay, where the agent learns from stored experiences in the replay buffer sometimes and other times from fresh experiences of the latest episode) ✔️
2. the long-term knowledge is distilled in the neural network ✔️
3. (the critical component, missing from current architectures): relearn old (compressed) knowledge in order to avoid catastrophic forgetting (CF) or ensure that this knowledge isn't forgotten

Catastrophic forgetting could occur for example when good policy experiences only 'good states' and theirfore unlearns what good behavior would look like in 'bad states' (e.g. the agent is about to crash into an obstacle). If the (good) policy then encounters a bad state that is not often experienced by the policy, this could lead to unpredictable (and probably bad) behavior following this state, e.g. you could imagine a car that slides too close to the edge of the road on wet ground and then doesn't know how to recover from this situation because it only knows the middle of the road, where it usually is. Or for a simple environment like cartpole, this would be a state where the pole is almost falling, and a good policy that usually only experiences states where it holds the pole upright wouldn't know how to recover properly. This phenomenon is the result of the replay buffer filling up with only 'good experiences' (no bad states like the cartpole almost tiping over are present in the buffer), because the bad (exploratory) experiences got pushed out of the buffer by newer experiences.

My idea now is to still keep some old experiences in order to mitigate this catastrophic forgetting (unlearning good behavior for states that are not encountered, because the policy doesn't select trajectories that lead to these states often). 
To do this efficiently, you could do one of the following:

- try to compress the old experiences to get some kind of basis for long-term memory (the agent continuously relearns from these old experiences)
- find a good heuristic for which old states should be kept inside the buffer and not thrown away like in normal (prioritized) experience replay buffers.

For that to work, we need to ensure that old (long-term) knowledge isn't forgotten by the network because the corresponding experiences got pushed out of the replay buffer by newer experiences. Theirfore our goal is to capture the entire data distribution in the replay buffer, so that the policy doesn't suffer from catastrophic forgetting, while not slowing down the learning of new knowledge by much.

### Proposed solutions

We can achieve this goal by compressing old experiences into new (much fewer!) experiences that the agents relearns from (if we compress multiple experiences into one and learn from this single datapoint multiple times, we have to be really careful about not overfitting to this single point), or we could introduce a new parameter to balance the tradeoff between throwing out old actions and having a diverse buffer (dataset).

- when learning from old policies, we have to use the probability of taking that action of current policy.

- we want to capture the entire distribution while favoring newer data, theirfore we need a mechanism that measures the similarity of new data compared to the old data and a bonus for freshness

- tradeoff between freshness of data and dis-similarity between that piece of data compared to the entire dataset (we can approximate the distribution of the dataset by the current entries of the replay buffer or by the knowledge compressed in the neural network (question: how could you do that?) ) (capturing edge-cases that might be more important to remember) After the replay buffer is filled up, we need a measure how similar a new experience/episode is to all experiences/episodes in the buffer.
We have to calculate all of the TD-Errors from the perspective of our current policy (for that, we use $V(s)$ and $V(s')$).


How can you measure the similarity of datapoints (experiences)?
- first idea: learn an embedding function and compare the vectors, e.g. by the cosine-similarity function

Experiences that want to throw away first:
- old data (> include an episode-index for each experience to know how fresh or old the experience is)
- data with a low TD-Error from the current policy
- data that is represented often in the dataset and theirfore has a high sum of similarities to each other datapoint in the buffer 

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
7. [Wiki: Sleep and leaning][wiki-sleep-learning]
8. [Wiki: Dreams][wiki-dreams]

<!-- Ressources -->
[RESSOURCE]: LINK
[endtoendai]: https://www.endtoend.ai/paper-unraveled/cer/
[endtoendai-fundamentals-of-ER]: https://www.endtoend.ai/one-slide-summary/revisiting-fundamentals-of-experience-replay/
[thecomputerscientist]: https://www.youtube.com/watch?v=MqZmwQoOXw4
[zhnang-sutton]: https://arxiv.org/abs/1712.01275
[prioritized-experience-replay]: https://arxiv.org/abs/1511.05952
[catastrophic-forgetting-dual-memory]: https://arxiv.org/pdf/1812.02464.pdf
[wiki-sleep-learning]: https://en.wikipedia.org/wiki/Sleep_and_learning
[wiki-dreams]: https://en.wikipedia.org/wiki/Dream

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