---
layout: post
title:  "Multishot Solving"
author: "Till Zemann"
date:   2023-01-08 14:31:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
tags: [uni, asp]
thumbnail: "/images/multishot_solving/robinhoodmultishot_5559.webp"
---


<div class="img-block" style="width: 500px;">
    <img src="/images/multishot_solving/robinhoodmultishot_5559.webp"/>
</div>

<!--
<em style="float:right">First draft: 2023-01-07</em><br>
-->

<!--
### Contents
* TOC
{:toc}
-->

### A simple ExampleApp

`instance.lp`
```py
num(1).
result(@my_func(N)) :- num(N).
```

`clingo_script.py`
```py
from clingo.symbol import Number
from clingo.control import Control

class ExampleApp:
    @staticmethod           # removes the self argument from my_func
    def my_func(x):
        x = x.number        # convert the input to an int
        return Number(x+2)
    
    def run(self):
        ctl = Control()
        ctl.load("instance.lp")
        ctl.ground(
            [
                ("base", [])
            ],
            context=self)
        ctl.solve(on_model=print)

if __name__ == "__main__":
    app = ExampleApp()
    app.run()
```

We can execute it via:
```bash
python clingo_script.py
```

<div class="output">
num(1) result(3)
</div>

### A clingo application

`clingo_app.py`
```py
import sys
from clingo.symbol import Number
from clingo.application import Application, clingo_main

class ExampleApp(Application):
    program_name = "example"
    version = "1.0"

    @staticmethod
    def my_func(x):
        x = x.number        # convert the input to an int
        return Number(x+2)
    
    def main(self, ctl, files):

        # load files into the clingo object
        for path in files:
            ctl.load(path)
        if not files:
            ctl.load("-")
        
        ctl.ground(
            [
                ("base", [])
            ],
            context=self)
        ctl.solve()

if __name__ == "__main__":
    app = ExampleApp()
    files = sys.argv[1:]
    clingo_main(app, files)
```

We can execute it via:
```bash
python clingo_app.py instance.lp
```

<div class="output">
num(1) result(3)
</div>


### Metaprogramming

Our `meta-telingo.lp` encoding gets a reified instance or program (_.lp file_) as input. Reification represents the truth of logical statements as an atom, rather than just assigning it a boolean value. We can then use these reified atoms to do all kinds of stuff (e.g. write our own temporal operators) with the statement.

To achieve the reification, we use the predicates `holds/2` and `true/1`.  



### meta-telingo

To use the meta-telingo.lp encoding, first go to the directory.
```bash
cd ~/Desktop/GitHub/telingo-wise22-23-themetaprogrammers/meta
```

To test it, reify a test instance using clingo and pipe it into meta-telingo. 
This is our tiny test instance:

```c
b :- finally.
until(a,b).

% solution: 
% 0: a
% 1: a
% 2: b

#external initially.
#external finally.
#external until(a,b).
#show show(a).
#show show(b).
```


To understand the _meta-programming_ process a little better, here is the output of the reification process of our `simple_test.lp` program:

<div class="output">
atom_tuple(0). <br>
atom_tuple(0,1). <br>
literal_tuple(0). <br>
rule(disjunction(0),normal(0)). <br>
external(1,false). <br>
external(2,false). <br>
external(3,false). <br>
atom_tuple(1). <br>
atom_tuple(1,4). <br>
literal_tuple(1). <br>
literal_tuple(1,2). <br>
rule(disjunction(1),normal(1)). <br>
output(finally,1). <br>
literal_tuple(2). <br>
literal_tuple(2,4). <br>
output(b,2). <br>
literal_tuple(3). <br>
literal_tuple(3,3). <br>
output(initially,3). <br>
output(until(a,b),0). <br>
literal_tuple(4). <br>
literal_tuple(4,-5). <br>
output(show(b),4). <br>
output(show(a),4). <br>
</div>



Now we can clingo to solve the `simple_test.lp` program. To do that, we are using the parameters `horizon=2` (which solves this instance) and set the number of models as `0`, which is a special case to show all models. The pipe (`|`) gives the atoms that are generated in the reification to the `meta-telingo.lp` encoding, which in return solves the temporal program.

```bash
clingo simple_test.lp --output=reify | clingo - meta-telingo.lp -c horizon=2 0
```

<div class="output">
clingo version 5.4.1 <br>
Reading from - ...<br>
Solving...<br>
Answer: 1<br>
(b,2) (a,0) (a,1)<br>
SATISFIABLE<br>
<p class="vspace"></p>
Models       : 1<br>
Calls        : 1<br>
Time         : 0.015s (Solving: 0.00s 1st Model: 0.00s Unsat: 0.00s)<br>
CPU Time     : 0.013s<br>
</div>

Great! Now let's try that with our iterative solving. To do this, we first need to convert our python script into a clingo file, so that clingo can use it. This can be simply done by adding `#script (python)` at the beginning and `#end.` at the end of the script.

But before we use the more complex script, we are going to start simple with a really small script to make debugging easier.

Here is the script:

```c
#script (python)

from clingo import Function, Number

def main(prg):
    prg.ground([("base", [])])
    prg.solve()

    step = 42
    parts = []
    parts.append(("step", [Number(step)]))
    prg.ground(parts)
    prg.solve()
#end.

#program base.

a.

#program step(t).

b(t).

#program check(t).
```

The program needs some input, so to run it we can just pipe a `test.` atom into it using `echo test.`:

```bash
echo test. | clingo - incremental_solver.lp -c horizon=2 0
```
<div class="output">
clingo version 5.4.1 <br>
Reading from - ... <br>
<p class="vspace"></p>
Solving... <br>
Answer: 1 <br>
a test <br>
<p class="vspace"></p>
Solving... <br>
Answer: 1 <br>
a test b(10) <br>
SATISFIABLE <br>
<p class="vspace"></p>
Models       : 2 <br>
Calls        : 2 <br>
Time         : 0.009s (Solving: 0.00s 1st Model: 0.00s Unsat: 0.00s) <br>
CPU Time     : 0.009s <br>
</div>


- https://arxiv.org/pdf/1705.09811.pdf
- https://potassco.org/clingo/python-api/current/clingo/





<hr><p class="vspace"></p>

More advanced stuff starts here (this is not finished yet!).

```bash
clingo simple_test.lp --output=reify | clingo - incremental_solver.lp -c horizon=2 0
``` 


You can find a more complete guide on the clingo API with examples [here](https://potassco.org/clingo/python-api/current/clingo/) (this should help when writing the python script).



### Iterative multishot solving

The idea is to only ground the base once (at the beginning) and then only ground the things that change with the horizon $t$ at each step to save some time on grounding. The base and step groundings are then combined to get the program for horizon $t$:

<svg width="800" height="200" version="1.1" xmlns="http://www.w3.org/2000/svg">
	<ellipse stroke="black" stroke-width="1" fill="none" cx="247.5" cy="57.5" rx="70" ry="30"/>
	<text x="192.5" y="63.5" font-family="Times New Roman" font-size="20">grounded base</text>
	<ellipse stroke="black" stroke-width="1" fill="none" cx="515.5" cy="57.5" rx="70" ry="30"/>
	<text x="450.5" y="63.5" font-family="Times New Roman" font-size="20">grounded step(t)</text>
	<ellipse stroke="black" stroke-width="1" fill="none" cx="377.5" cy="151.5" rx="145" ry="30"/>
	<text x="259.5" y="157.5" font-family="Times New Roman" font-size="20">combined grounded program</text>
	<ellipse stroke="black" stroke-width="1" fill="none" cx="377.5" cy="151.5" rx="150" ry="35"/>
	<polygon stroke="black" stroke-width="1" points="271.811,75.078 353.189,133.922"/>
	<polygon fill="black" stroke-width="1" points="353.189,133.922 349.636,125.182 343.777,133.286"/>
	<polygon stroke="black" stroke-width="1" points="490.706,74.389 402.294,134.611"/>
	<polygon fill="black" stroke-width="1" points="402.294,134.611 411.721,134.24 406.091,125.975"/>
</svg>


### Parallel multishot solving



### Todo

- write a section on the meta-programming paradigm


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
1. Thumbnail taken from [here][thumbnail].
2. [Martin Gebser, Roland Kaminski, Benjamin Kaufmann and Torsten Schaub: Multi-shot ASP solving with Clingo][multishot-solving-paper]


<!-- Ressources -->
[thumbnail]: https://static.tvtropes.org/pmwiki/pub/images/robinhoodmultishot_5559.jpg
[multishot-solving-paper]: https://arxiv.org/pdf/1705.09811.pdf

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
