---
layout: post
title:  "Transformers"
author: "Till Zemann"
date:   2023-02-18 01:09:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
positive_reward: true
reward: 2
tags: [machine learning, nlp]
thumbnail: "/images/transformers/architecture.png"
---

<!--
### Contents
* TOC
{:toc}
-->

<div class="img-block" style="width: 400px;">
    <img src="/images/transformers/Attention_is_all_you_need.jpg"/>
</div>

<em style="float:right">First draft: 2023-02-20</em><br>


### Introduction

The Transformer was invented in the "Attention is all you need" paper in 2017 and has held the state of the art title in Natural Language Processing for the last 5-6 years. It can be applied to any kind of sequential task and is successful in a lot of domains and with many variations (although the architecture presented in the 2017 paper still pretty much still applies today).

The basic idea of the Transformer is to build an architecture around attention-functions. Multiplicative attention blocks (where the weighting factors of the values are calculated by a dot-product between queries and keys) can be parallelized and become super fast. This is one of the major advantages over all types of RNNs, where the input has to be processed sequentially. Another advantage is the number of processing steps between inputs that are multiple timesteps apart: for RNNs capturing long-range dependencies is really difficult and with Transformers, the inputs are related in a constant number of processing steps and theirfore even long-range dependencies can be captured pretty easily using attention.

<center>
	The code for this post is available at <a href="https://github.com/till2/GPT_from_scratch">https://github.com/till2/GPT_from_scratch.</a>
</center>
<p class="vspace"></p>



### Goal of this blogpost

The goal for this post is to build a encoder-only transformer. The left side of the transformer in the picture is the encoder, which you would use for tasks that require additional information that first needs to be encoded, an example would be a translation task, where you first have to encode the sentence in the original language using the encoder. Then you can use the decoder to process the growing sequence of tokens in the target language.

We'll only use the encoder part with self-attention, because we just want to have a text-block as context (prompt) and complete it (meaning we can generate more text that fits the given prompt) and theirfore don't need both parts.

<!-- Architecture -->

<!--
<div class="img-block" style="width: 950px;">
    <img src="/images/transformers/transformer.png"/>
</div>
-->

### Architecture

For reference how each part that we'll talk about is integrated, here is a complete view of the encoder-only transformer architecture:

<div class="img-block" style="width: 800px;">
    <img src="/images/transformers/architecture.png"/>
</div>



### Encodings

#### Character-level Encoding

Character-encoding just maps every unique character in the text corpus to an integer. With punctuation, digits, letters (including some chinese I believe) the Lex Fridman podcast turns out to include around 150 unique characters. As you probably can imagine, this is the easiest form of tokenization to implement.


#### Byte-Pair Encoding (BPE)

BPE is used very commonly in practice, for example for the GPT models by OpenAI. It is a form of sub-word tokenization and combines the advantages of giving the model easier access to common sub-words (the byte pairs), which should make it easier to generate comprehensive language, as well as the ability to also understand uncommon words more intuitively through their sub-words (as you can split them into their prefixes, suffixes etc., which should be more common sub-words).


Here are the 150 (approximately) most common byte pairs from my dataset: 
<div class="output">
t , e , m, th, , an, ,, in, s , ou, d , ve, l , the , at , er, li, y , g , es, so, r , to, xes, I , . , ma, ive, you, ri, f , you , be, me, that , in, to , in , I', do, wh, ll, you', er , ing , ch, k , mo, t c, what , iver, ves, pe, ow, out , ro, , es , ere , 's , . A, le , us, of , t's , fo, d the , t in, rs, mag, co, n , st, re, ing to , per, now, ant, and , y o, an , , you , ke, do , , you k, cou, me , ink , ha, stu, pro, mat, ke , foo, ry , , and , of the , d to , you w, don, ver , he , men, go, ? , jus, mati, have, n a, wi, gove, res, ros, bu, lly , 're , some, . An, , and , use , on , we , on the , . H, ? , I w, t to , s that , ion , ll , be , with, No, pet, t of , perso, sou, . And , mbe, the w, . He , have , tog, just , re p, ment, les, like , going , you're , mod
</div>



### Embeddings

#### Token embedding

Every token has a learned embedding in the token embedding table.

### Positional embedding

The position is embedded using a learned second embedding table.
The token-embedding and position-embedding matrices are added to get the input for the transformer.


### Attention

I find attention very intuitive to understand when you look at it from a database perspective:
You have your <strong style="color: #1E72E7">query (Q)</strong>, <strong style="color: #ED412D">key (K)</strong> and <strong style="color: #747a77">value (V)</strong> vectors and want to weight the values according to how much a query matches with every value. If you have a database lookup with one hit, the query-key pair for the hit would result in a weight of 1 and every other query-key pair would result in a weight equal to 0, so only the value for that matching key gets returned.

We use vector products to do this weighting, and thus can have more than one query. Actually we can process a vector <strong style="color: #1E72E7">Q</strong> of queries that gets matched against keys to create a weight matrix where row $i$ corresponds to the weights for the corresponding query $q_i$.


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

<p class="vspace"></p>

To illustrate the difference that scaling the weight-logits has on the gradient, I took the gradient $\frac{\partial z}{\partial z_0}$ of the vector $z = [z_0, z_1, z_2, z_3] = [1, 20, 3, 4]$.
The resulting gradient is already tiny with just one larger number (the 20 at index 1):

__Tiny gradient without scaling:__
<div class="output">
[5.6e-09, -5.6e-09, -2.3e-16, -6.3e-16]
</div>

Here $d_k = 4$, so we multiply the matrix $X$ by the scaling factor $\frac{1}{\sqrt{4}}$ to get the scaled $X$. If we take the gradient now, it's not as small as the previous gradient. This way our network can learn quicker.

__Better gradient via scaling:__
<div class="output">
[0.0001, -2.7149, -1.1240e-07, -3.0553]
</div>


The resulting weight matrix <strong>W</strong> can finally be multiplied with the value vector <strong style="color: #747a77">V</strong> to get the output of one (scaled dot-product) attention unit:

$$
\text{Attention}(Q,K,V) = WV = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

<p class="vspace"></p>

### Implementing Attention


First, let's create some random vectors for <strong style="color: #1E72E7">Q</strong>, <strong style="color: #ED412D">K</strong> and <strong style="color: #747a77">V</strong> with shape $5 \times 1$.
```py
Q = torch.rand(5,1)
K = torch.rand(5,1)
V = torch.rand(5,1)
```

We get the logits for the weight matrix by using the scaled query-key matrix multiplication. In PyTorch, we can use the `@`-Symbol to perform a matrix multiplication.

```py
C, B = Q.shape # Channels and Batch-Size

W = (Q @ K.T) / torch.sqrt(torch.tensor(C))
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

To get the final weight matrix, we apply a softmax on each row. Each row now sums up to 1.
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

In our actual implementation we use the attention head as a class which also already includes the linear projections for the query, key and value vectors:
```py
class SelfAttentionHead(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.proj_q = nn.Linear(embed_dims, head_size, bias=False)
        self.proj_k = nn.Linear(embed_dims, head_size, bias=False)
        self.proj_v = nn.Linear(embed_dims, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """ 
        Applies masked scaled dot-product attention
        between vectors of queries Q, keys K and values V. 
        """
        B,T,C = x.shape
        
        Q = self.proj_q(x)
        K = self.proj_k(x)
        V = self.proj_v(x)

        W = (Q @ K.transpose(-1,-2)) # (B, T, C) @ (B, C, T) ==> (B,T,T)
        W /= torch.sqrt(torch.tensor(head_size))
        
        # mask out forbidden connections
        tril = torch.tril(torch.ones((block_size, block_size), device=device))
        W = W.masked_fill(tril[:T, :T]==0, float("-inf")) # make smaller so it fits if context < block_size
        W = F.softmax(W, dim=1)
        W = self.dropout(W)
        
        out = W @ V
        return out # (B,T,C=head_size)
```
### Masking

Notice that a self-attention head masks its inputs.

We want a position to only attent to the past in the decoder block, because they won't have future tokens available yet and thus can't learn to attent to the future. Each token can only attent to it's own position and all past positions in the given context sequence.
To mask all attention connections to the future out, we use a lower triangular matrix (_note: tril means triangular-lower_).

```py
T = 10
tril = torch.tril(torch.ones((T,T)))
plt.imshow(tril)
```

<div class="img-block" style="width: 400px;">
    <img src="/images/transformers/mask.png"/>
</div>

<center>The yellow is all 1s and the purple all 0s (0 means that the connection is not allowed).</center>
<p class="vspace"></p>

```py
W = torch.rand((T,T)) # there will be real data here

# mask out forbidden connections
W = W.masked_fill(tril==0, float("-inf")) # set everywhere where tril is 0 to -inf (upper right)

W = F.softmax(W, dim=-1)
plt.imshow(W)
```

<div class="img-block" style="width: 400px;">
    <img src="/images/transformers/mask2.png"/>
</div>


### Multi-Head Attention


#### Linear Projection

In multi-headed attention, we apply multiple attention blocks in parallel. To encourage that they learn different concepts, we first apply linear transformation matrices to the <strong style="color: #1E72E7">Q</strong>, <strong style="color: #ED412D">K</strong>, <strong style="color: #747a77">V</strong> vectors. You can intuitively look at this as viewing the information (vectors) from a different angle.

To get an idea about how this looks, here is a simple linear transformation of the unit vector $v = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ in 2D space.

<table style="background-color: green;">
  <tr>
    <th>
<div style="color:#505050;">
$$
A = \begin{bmatrix} -0.7 & 1 \\ 1 & -0.2 \end{bmatrix} \\ 
$$
</div>
    </th>
    <th>
<div style="color:magenta;">
$$
v = \begin{bmatrix} 1 \\ 1 \end{bmatrix} \\ 
$$
</div>
    </th>
    <th>
<div style="color:blue;">
$$
Av = \begin{bmatrix} 0.3 \\ 0.8 \end{bmatrix}\\
$$
</div>
    </th>
  </tr>

</table> 

<div class="img-block" style="width: 400px; ">
    <img src="/images/transformers/linear_proj.png"/>
</div>



We can simply implement these linear projections as a Dense layer without any biases. The weights of the projections can be learned so that the Transformer uses the most useful projections of the Q,K,V vectors. A useful property of these projections is that we can pick the number of dimensions for the space that they are projected into, which usually has a lower dimensionality than the input vectors so that it is computationally feasible to have multiple heads running in parallel (you want to set it so that your GPUs VRAM is maximally utilized).


To implement multi-head attention, we first define linear layers for each head and for each Q,K,V vector. We also need a linear layer that combines the output of all parallel attention blocks into one output vector.

```py
class MultiHeadAttention(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.heads = nn.ModuleList([SelfAttentionHead() for i in range(n_heads)])
        self.proj = nn.Linear(embed_dims, embed_dims, bias=False) # embed_dims = n_heads * head_size
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        
        out = torch.cat([attn_head(x) for attn_head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
```


### Add & Norm and Residual Connections

We use pre-layernorm [(performs bettern than post-layernorm)][pre-ln-paper], which is different from the original transformer.
What we keep is the residual connections around the multi-head self-attention and around the mlp (simple feed-forward network with 2 dense layers and relu activations).

A transformer block also includes a feed forward network, so that it follows these two stages:

1. Communicate via self-attention
2. Process the results using the MLP


```py
class Block(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.ln1 = nn.LayerNorm(embed_dims)
        self.ln2 = nn.LayerNorm(embed_dims)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dims, 4*embed_dims), # following attention-is-all-you-need paper for num hidden units
            nn.ReLU(),
            nn.Linear(4*embed_dims, embed_dims),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        
        # Applies layernorm before self-attention.
        # In the attention-is-all-you-need paper they apply it afterwards, 
        # but apparently pre-ln performs better. pre-ln paper: https://arxiv.org/pdf/2002.04745.pdf
        
        x = x + self.attn(self.ln1(x)) # (B,embed_dims)
        x = x + self.mlp(self.ln2(x))
        return x
```



### Training

<div class="img-block" style="width: 600px;">
    <img src="/images/transformers/training_plot1.png"/>
</div>


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
0. [Thumbnail](https://makeameme.org/meme/attention-please-d7217f13d3)
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
11. [AI Coffee Break with Letitia: Positional embeddings in transformers EXPLAINED - Demystifying positional encodings.](https://youtu.be/1biZfFLPRSY)
12. [Ruibin Xiong et. al: On Layer Normalization in the Transformer Architecture][pre-ln-paper] (Pre-LayerNorm paper)

<!-- Ressources -->
[transformer-img]: https://deepfrench.gitlab.io/deep-learning-project/resources/transformer.png
[transformer-paper-2017]: https://arxiv.org/pdf/1706.03762.pdf
[the-annotated-transformer]: https://nlp.seas.harvard.edu/2018/04/03/attention.html
[rasa-self-attention-video]: https://youtu.be/yGTUuEx3GkA
[pre-ln-paper]: https://arxiv.org/pdf/2002.04745.pdf

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