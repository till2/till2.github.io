---
layout: post
title:  "Astrophysics"
author: "Till Zemann"
date:   2023-03-06 01:08:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
positive_reward: true
reward: 5
tags: [physics, long read]
thumbnail: "/images/astrophysics/SN1994D.jpg"
---

<!--
### Contents
* TOC
{:toc}
-->

<div class="img-block" style="width: 400px;">
    <img src="/images/astrophysics/SN1994D.jpg"/>
</div>

<!--
<em style="float:right">First draft: 2023-02-20</em><br>
-->


### Contents
* TOC
{:toc}


### Definitions and random Notes

- 1° = 3600'' (arcseconds)
- Symbol for the sun: $\odot$
- Symbol for a star: $\*$
- semi-major axis of an ellipse is called $a$


### Introduction

#### asdf

##### adsf



### Geometry

#### Circle

Surface Area $A = \pi R^2$

Circumference $u = 2 \pi R$

Diameter $d = 2 R$

#### Sphere

Volume $V = \frac{4}{3} \pi R^3$

Surface Area $A = 4 \pi R^2$

#### Ellipse

- approximate by using formulas for a circle
- semi-major axis is called $a$:

<div class="img-block" style="width: 300px;">
    <img src="/images/astrophysics/ellipse.png"/>
</div>


### VL2 - Kepler orbits and the 2-body problem

- https://de.wikipedia.org/wiki/Zweik%C3%B6rperproblem

### VL3 - Our Sun



### VL4 - Our Solar system


### VL5 - Stars



### VL6 - The Life Cycle of Cosmic Material

- https://www.theexpertta.com/book-files/OpenStaxAstronomy/Astronomy_20.5.%C2%A0The%20Life%20Cycle%20of%20Cosmic%20Material_pg714%20-%20716.pdf

### VL7 - Exoplanets

- Exoplanet means extrasolar planet (planet outside the solar system)

#### Relative Luminosity of a Planet to its Star

$$
\frac{s_P}{S_*} \approx A \cdot 0.5 \cdot \frac{\pi R_P^2}{4 \pi a_P^2}
$$

with the albedo $A \in [0.2, 0.8]$, the fraction of a planet that is lit by its star (always half)



### VL8 - The Milky Way and Galaxies

- https://en.wikipedia.org/wiki/Milky_Way

### VL9 - Dark Matter


### VL10 - Black Holes




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
0. [Thumbnail: Galaxy NGC 4526 with Supernova 1994D in the bottom left corner, distance=50M ly, constellation: virgo (credit: ESA/Hubble, released: 25 May 1999)][thumbnail-galaxy-supernovae]


<!-- Ressources -->
[thumbnail-galaxy-supernovae]: https://esahubble.org/images/opo9919i/



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