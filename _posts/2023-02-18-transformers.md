---
layout: post
title:  "The Transformer"
author: "Till Zemann"
date:   2023-02-18 01:09:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
positive_reward: true
reward: 2
tags: [machine learning, nlp, not finished yet]
thumbnail: "/images/transformers/transformer.png"
---

<!--
### Contents
* TOC
{:toc}
-->

### Introduction

The Transformer was invented in the "Attention is all you need" paper in 2017 and has held the state of the art title in Natural Language Processing for the last 5-6 years. It can be applied to any kind of sequential task and is successful in a lot of domains and with many variations (although the architecture presented in the 2017 paper still pretty much still applies today).

The basic idea of the Transformer is to build an architecture around attention-functions. Multiplicative attention (where the weighting factors of the values are calculated by a dot-product between queries and keys) can be parallelized amazingly well. This is one of the major advantages over all types of RNNs, where the input has to be processed sequentially. Another advantage is the number of processing steps between inputs that are a number of timesteps apart: for RNNs capturing long-range dependencies is really difficult and with Transformers, the inputs are related in a constant number of processing steps and theirfore even long-range dependencies can be captured pretty easily using attention.


### Attention


I find attention very intuitive to understand when you look at it from a database perspective:
You have your <strong style="color: #1E72E7">query (Q)</strong>, <strong style="color: #ED412D">key (K)</strong> and <strong style="color: #747a77">value (V)</strong> vectors and want to weight the values according to how much a query matches with every value. If you have a database lookup with one hit, the query-key pair for the hit would result in a weight of 1 and every other query-key pair would result in a weight equal to 0, so only the value for that matching key gets returned.

Because we can just use vector products to do this weighting, we can not just have one query, but a vector of queries that gets matched against keys to create a weight matrix where row $i$ corresponds to the weights for the corresponding query $q_i$.


<div class="img-block" style="width: 800px;">
    <img src="/images/transformers/key_query_value.png"/>
</div>


#### Scaling

Actually this is not enough to get proper weighting factors, because the weights should sum to 1. To achieve this, we can apply a softmax per row (dim=1).
In the "Attention is all you need" paper, they also scale the matrix by a factor of $\frac{1}{\sqrt{d_k}}$. If you don't do that, the gradient for large arguments of the softmax function is tiny and theirfore the entire architecture doesn't learn as quickly without scaling. 

<div class="img-block" style="width: 800px;">
    <img src="/images/transformers/weights.png"/>
</div>


The gradient of the softmax function is: <br>

$$
\frac{\partial \text{ S}(z_i)}{\partial z_j} = 
\begin{cases}
S(z_i)(1-S(z_i)) 	& \text{ if } i=j \\
-S(z_i)S(z_j) 		& \text{ if } i \neq j\\
\end{cases}
$$

To illustrate the difference, I took the gradient $\frac{\partial z}{\partial z_0}$ of the vector $z = [z_0, z_1, z_2, z_3] = [1, 20, 3, 4]$.
The resulting gradient is already tiny with just one larger number (the 20 at index 1):

<div class="output">
[5.6e-09, -5.6e-09, -2.3e-16, -6.3e-16]
</div>

Here $d_k = 4$, so we multiply the matrix $X$ by $\frac{1}{\sqrt{4}} = \frac{1}{2}$ to get the scaled $X$. In our example $d_k = 4$, so we multiply X by $\frac{1}{\sqrt{4}} = \frac{1}{2}$. If we take the gradient now, it's not as small as the previous gradient. This way our network can learn quicker.

<div class="output">
[0.0001, -2.7149, -1.1240e-07, -3.0553]
</div>

You can use this code-snipped to try it out yourself:
<script src="https://gist.github.com/till2/e1a554b6d41c4d2d4f266180827ffd9a.js"></script>


The resulting weight matrix <strong>W</strong> can finally be multiplied with the value vector <strong style="color: #747a77">V</strong> to get the output of one (scaled dot-product) attention unit:

$$
\text{Attention}(Q,K,V) = WV = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$


### Implementing Attention


First, let's create some random vectors for <strong style="color: #1E72E7">Q</strong>, <strong style="color: #ED412D">K</strong> and <strong style="color: #747a77">V</strong> with shape $5 \times 1$.
```py
Q = torch.rand(5,1)
K = torch.rand(5,1)
V = torch.rand(5,1)
```

We get the logits for the weights by using the scaled query-key matrix multiplication. In PyTorch, we can use the `@`-Symbol to perform a matrix multiplication.

```py
d_k = torch.tensor(Q.shape[0]) # 5

W = (Q @ K.T) / torch.sqrt(d_k)
W
```

<div class="output">
tensor([<br>
[0.0539, 0.2369, 0.0690, 0.1660, 0.1998],<br>
[0.0256, 0.1125, 0.0328, 0.0788, 0.0949],<br>
[0.0198, 0.0870, 0.0254, 0.0610, 0.0734],<br>
[0.0781, 0.3433, 0.1000, 0.2405, 0.2895],<br>
[0.0246, 0.1081, 0.0315, 0.0757, 0.0912]])
</div>

To get the final weight matrix, we apply a softmax on each row (each row sums to 1).
```py
W = F.softmax(W, dim=1)
W
```

<div class="output">
tensor([<br>
[0.1964, 0.2037, 0.1970, 0.2007, 0.2021],<br>
[0.1983, 0.2018, 0.1986, 0.2004, 0.2010],<br>
[0.1987, 0.2014, 0.1989, 0.2003, 0.2008],<br>
[0.1949, 0.2055, 0.1956, 0.2010, 0.2030],<br>
[0.1984, 0.2017, 0.1986, 0.2004, 0.2010]])
</div>

Now we just have to apply our computed weight-matrix to our values to get the final attention output. 
```py
W @ V
```

<div class="output">
tensor([<br>
[0.4725],<br>
[0.4733],<br>
[0.4734],<br>
[0.4719],<br>
[0.4733]])
</div>


#### Concise Implementation of Scaled Dot-Product Attention

Putting it all together, we get:

```py
def attention(Q,K,V):
    """ 
    Applies scaled dot-product attention
    between vectors of queries Q, keys K and values V. 
    """
    d_k = torch.tensor(Q.shape[0])
    W =  F.softmax((Q @ K.T) / torch.sqrt(d_k), dim=1)
    return W @ V
```



### Multi-Head Attention


#### Linear Projection

In multi-headed attention, we perform multiple attention blocks in parallel. To encourage that they learn different concepts, we first apply linear transformation matrices to the <strong style="color: #1E72E7">Q</strong>, <strong style="color: #ED412D">K</strong>, <strong style="color: #747a77">V</strong> vectors. You can intuitively look at this as viewing the information (vectors) from a different angle.

To get an idea about how this looks, here is a simple linear transformation of the unit vector $v = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ in 2D space.

<div style="color:grey;">
$$
A = \begin{bmatrix} -0.7 & 1 \\ 1 & -0.2 \end{bmatrix} \\ 
$$
</div>

<div style="color:magenta;">
$$
v = \begin{bmatrix} 1 \\ 1 \end{bmatrix} \\ 
$$
</div>

<div style="color:blue;">
$$
Av = \begin{bmatrix} 0.3 \\ 0.8 \end{bmatrix}\\
$$
</div>

<div class="img-block" style="width: 400px;">
    <img src="/images/transformers/linear_proj.png"/>
</div>


We can simply implement these linear projections as a Dense layer without any biases. The weights of the projections can be learned so that the Transformer uses the most useful projections of the Q,K,V vectors. A useful property of these projections is that we can pick the number of dimensions for the space that they are projected into, which usually has a lower dimensionality than the input vectors so that it is computationally feasible to have multiple heads running in parallel (you want to set it so that your GPUs VRAM is maximally utilized).


To implement multi-head attention, we first define linear layers for each head and for each Q,K,V vector. We also need a linear layer that combines the output of all parallel attention blocks into one output vector.

```py
def multi_head_attention(Q,K,V):
    d_k = torch.tensor(Q.shape[0])
    d_model = 8 # project in to this space
    N_heads = 2
    
    # linear layers
    projections = {
        x: {
            h: nn.Linear(d_k, d_model, bias=False) for h in range(N_heads)
        } for x in ["Q", "K", "V"]
    }
    
    # layer to combine the concatenated attention-block output vectors
    top_layer = nn.Linear(N_heads * d_model, d_k, bias=False)
    
    # forward pass
    result = torch.zeros(N_heads, d_model, 1)

    for h in range(N_heads):
        result[h] = attention(
            projections["Q"][h](Q.T).T,
            projections["K"][h](K.T).T,
            projections["V"][h](V.T).T
        )
    
    concat_attn_out = result.view(1, N_heads * d_model)
    return top_layer(concat_attn_out).T
```

```py
multi_head_attention(Q,K,V)
```

<div class="output">
tensor([<br>
[-0.0233], <br>
[ 0.0191],<br>
[-0.0144],<br>
[ 0.0373],<br>
[-0.0110]]) <br>
</div>


### Add and Norm










### Transformer Architecture

<!-- Architecture -->

<div class="img-block" style="width: 800px;">
    <img src="/images/transformers/transformer.png"/>
</div>



### Positional encoding


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
2. Vaswani et. al: Attention Is All You Need - [Paper][transformer-paper-2017]
3. Rasa: [Rasa Algorithm Whiteboard - Transformers & Attention 1: Self Attention][rasa-self-attention-video] (This is the first video of a 4-video series about the Transformer, I can highly recommend it!)
4. [Andrej Karpathy:  Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) (Implementation)
5. [harvardnlp - The Annotated Transformer][the-annotated-transformer]
6. [Aleksa Gordić: Attention](https://www.youtube.com/watch?v=n9sLZPLOxG8) // <- this is part 1
7. [Aleksa Gordić: Transformer](https://www.youtube.com/watch?v=cbYxHkgkSVs) // <- and part 2
8. [11-785 Deep Learning Recitation 11: Transformers Part 1](https://www.youtube.com/watch?v=X2nUH6fXfbc) (Implementation)
9. [TensorFlow Blog: A Transformer Chatbot Tutorial with TensorFlow 2.0](https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html)
10. [Kaduri's blog: From N-grams to CodeX (Part 2-NMT, Attention, Transformer)](https://omrikaduri.github.io/2022/10/22/From-N-grams-to-CodeX-Part-2.html)

<!-- Ressources -->
[transformer-img]: https://deepfrench.gitlab.io/deep-learning-project/resources/transformer.png
[transformer-paper-2017]: https://arxiv.org/pdf/1706.03762.pdf
[the-annotated-transformer]: https://nlp.seas.harvard.edu/2018/04/03/attention.html
[rasa-self-attention-video]: https://youtu.be/yGTUuEx3GkA

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