---
layout: page
title: Why RL?
permalink: /why-rl/
---

<div class="img-block" style="width: 300px;">
    <img src="/images/why-rl/robotic-manipulation.gif"/>
</div>

<em style="float:right">First draft: 2022-12-16</em><br>

Reinforcement learning (RL) is a fascinating subject to study because it represents a significant step forward in our ability to create intelligent machines.

One of the primary reasons that reinforcement learning is so fascinating is that it allows machines to learn through trial and error, much like a human would. Instead of being explicitly programmed with a set of rules or instructions, a reinforcement learning agent is given a goal and a set of possible actions, and it learns to achieve that goal through interaction with an environment. Through learning the agent can adapt and improve its performance over time, as it gathers more experiences and learns how to chose good actions to achieve tasks.

<em> The section above was written by a Transformer (GPT3) tuned with Reinforcement Learning (GPT -> ChatGPT), demonstrating the ability of RL for text generation. </em> [[6]][chatgpt] <br> 
<em> (Prompt: "Write a section about why reinforcement learning is fascinating and awe-inspiring") </em> 


Another reason that reinforcement learning is awe-inspiring is that it has the potential to revolutionize a wide range of fields, from robotics and autonomous vehicles to healthcare, game playing, natural language processing, finance, education and tutoring, and many more. By enabling machines to learn and adapt to new situations and environments, reinforcement learning could lead to significant advancements in a variety of industries, improving efficiency and productivity and potentially even enabling or producing creativity.


The potential of RL extends further to creative solutions that can emerge for example through a combined approach of self-play with reinforcement learning. When an agent is given the freedom to explore and experiment with different actions and strategies in order to achieve a goal, typically in a competetive setting where it is playing against itself, the learning can lead to innovative, original and clever solutions, that even the builders of the system didn't anticipate.

<p class="vspace"></p>

### __A few examples of creative solutions <br> that emerged through RL:__

<p class="vspace"></p>

#### OpenAI hide and seek (agents learned to abuse the physics engine)
<!-- width: 560, height: 315-->
<iframe width="355" height="200" src="https://www.youtube.com/embed/Lu56xVlZ40M?start=240" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<iframe width="355" height="200" src="https://www.youtube.com/embed/kopoLzvh5jY?start=110" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<em>Videos are playing from the timestamps that show the strategies to exploit physics bugs.</em>
<p class="vspace"></p>
<hr style="margin-bottom:30px">

#### OpenAI gym ant walking upside down

<div class="img-block" style="width:355px;float:left">
    <img src="/images/why-rl/ant_walking_gate.gif"/>
</div>

<div class="img-block" style="width:355px;float:right">
    <img src="/images/why-rl/ant-headspins.gif"/>
</div>

__Left:__ normal walking pattern [[2]][ant-walking-gait]<br>
__Right:__ something similar to the discovered <br>head-walking gait (read text below) [[3]][ant-headspins]

OpenAI gym coupled with the MuJoCo simulator offers realistic physics simulation, so you can try to train agents to walk or do all kinds of stunts. Usually when learning a walking gait (pattern), you want to give some additional reward when the agent is walking smoothly (because otherwise it often jitters and looks really weird). So in one example someone wanted to make the ant jump forward instead of walking, and theirfore gave it negative reward for touching the ground. But instead of jumping, it discovered to walk on its head and moved forward on the knees, so the feet didn't touch the ground. Unfortunately I couldn't find the video for this, but the clip on the right should give you the gist of how that would look like. 
<p class="vspace"></p>
<hr style="margin-bottom:30px">

#### Similarly to the ant example, here we want to learn to run

Just watch for yourself in how many ways that can go wrong :D

<iframe width="560" height="315" src="https://www.youtube.com/embed/rhNxt0VccsE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<p class="vspace"></p>
<hr style="margin-bottom:30px">


#### Breakout tunnel

<em> Below, I have linked to the specific parts of the following videos that correspond to the text. However, if you find the video interesting, I highly recommend watching the entire thing. </em> 

<iframe width="560" height="315" src="https://www.youtube.com/embed/d-bvsJWmqlc?start=913" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<em>Agent discovering a creative strategy (tunnel digging) to beat breakout quickly. <br> (Wait for it.)</em>
<p class="vspace"></p>
<hr style="margin-bottom:30px">

#### The creative move 37 from AlphaGo, refuting thousands of years of Go theory

<iframe width="560" height="315" src="https://www.youtube.com/embed/WXuK6gekU1Y?start=2974" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<p class="vspace"></p>
<hr style="margin-bottom:30px">


Apart of creating insight, efficiency and creative solutions, the study of reinforcement learning also allows us to gain a deeper understanding of how intelligence works, both in humans and in machines. I find the thinking of Rich Sutton [[4] (with timestamp; <- this is probably the most important link in this post)][tea-talk-with-richard-sutton] appealing, who says we should keep an open mind about intelligence and not predetermine that human intelligence is an optimal way of going about it. Because it's not. Machines can already do a lot of intelligent things better than we can. I think there is still a big Delta of wiggle room before we create a system that resembles optimal intelligence, Artificial General Intelligence (AGI), or anything close to that.

By examining the learning systems, we can gain insights into the underlying mechanisms of learning and decision-making, and use this knowledge to:

1. design more intelligent systems in the future <br>
2. understand ourselves better

Overall, the study of reinforcement learning is a fascinating and awe-inspiring field that has the potential to shape the future of artificial intelligence and the way we interact with machines. 

And its results are sometimes just plain beautiful.

So i'm in.

### Todo

Ressources:
- david silver lectures for an introduction to rl
- sutton & barto (free online)

- add GIFS for RL agents playing

- building, analyzing and understanding intelligence 
- super fascinating because it resembles how humans learn
-> machines will be able to do it more efficiently/better

- AGI

-> help solve problems
-> come up with new theorems and proofs
-> help propose solutions to hard problems, e.g. mitigate/reduce climate change



- learning! (no.1. fact-based is limited by the knowledge put inside the system and the designers)
- attention
- curiosity

### Accomplishments

- add images for accomplishments

- ChatGPT (PPO + RLHF)
- AlphaGo (Deepmind-Go)
- AlphaZero (Deepmind-Any 2 player game)
- Google Datacenters (Deepmind)
- learning to drive in a day
- AlphaStar
- walking robodogs


### References

1. [OpenAI: Emergent Tool Use from Multi-Agent Interaction][openai-emergent-tool-use]
2. [Berkeley - normal ant walking gait][ant-walking-gait]
3. [Stuart Robinson with Isaac Sim - ant head spinning][ant-headspins]
4. [Rich Sutton - The Alberta Plan for AI Research: Tea Time Talk with Richard S. Sutton ][tea-talk-with-richard-sutton]
5. [Thumbnail taken from this Google blog][robotic-manipulation]
6. [ChatGPT - GPT with RL from human feedback][chatgpt]

<!-- Ressources -->
[openai-emergent-tool-use]: https://openai.com/blog/emergent-tool-use/
[ant-walking-gait]: https://bair.berkeley.edu/static/blog/model-rl/fig_4c.gif
[ant-headspins]: https://learningreinforcementlearning.com/ant-antics-3566934125fb
[tea-talk-with-richard-sutton]: https://youtu.be/iS7dRTge8Z8?t=305
[robotic-manipulation]: https://ai.googleblog.com/2018/06/scalable-deep-reinforcement-learning.html
[chatgpt]: https://openai.com/blog/chatgpt/