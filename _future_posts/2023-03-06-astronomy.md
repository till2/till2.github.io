---
layout: post
title:  "Astronomy"
author: "Till Zemann"
date:   2023-03-06 01:08:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
tags: [physics]
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

<!--
Alternative title:
Astrophysics

Tags:
uni, physics, long read
-->

<!--
### Contents
* TOC
{:toc}
-->



## Quick Facts

### Notation

- Symbol for the sun: $\, \odot$
- Symbol for a star: $\, \*$
- The semi-major axis of an ellipse is called $\, a$
- Symbol for the earth: $E$ (i.e. the mass of the earth is written as $\, m_E$)


### Our Sun
Mass of the sun: $M_\odot = 1.989 \cdot 10^{24} \, \text{kg}$ \\
Volume of the sun: $V_\odot = 1.41 \cdot 10^{18} \, \text{km}^2$ \\
Luminosity of the sun: $\mathcal{L_\odot} = 3.828 \cdot 10^{26} \, \text{W}$ \\
Lifetime of the sun: $t_\odot = 10 \cdot 10^{9} \, \text{years}$


### Useful constants

Gravitational constant: $G = 6.67 \cdot 10^{-11} \, \frac{\text{m}^3}{\text{kg} \cdot s^2}$ \\
Stefan Boltzmann constant: $\sigma_{SB} = 5.67 \cdot 10^{-8} \, \frac{\text{N}}{\text{m}^2 \cdot \text{K}^4}$


### Useful conversions


- Degrees to arcseconds: $1° = 3600''$
- $1$ year = $365.25$ days
- Degrees Kelvin to Celsius: $X$° K - 273.15 = $X$° C
- Parsec to light years: $1 \, \text{pc} \approx 3.26 \, \text{ly} $
- Astronomical unit: $1 \text{AU} = 149 597 870 700 \, \text{m} \approx 1.5 \cdot 10^{11} \, \text{m} \approx 4.85 \cdot 10^{-6} \, \text{pc}$
- $sin(\alpha) = \frac{\text{opposite cathetus}}{\text{hypotenuse}}$
- Joules to mega electron volts: $1 \text{J} \approx 6.24 \cdot 10^{12} \, \text{MeV}$
- Cubic centimeters to cubic meters: $1 \, \text{cm}^3 = 1 \cdot 10^{-6} \, \text{m}^3$ (reduce exponent by 6) 



### Geometric refresher


| Object  | Formulas                                                                                                                                                                                                                         |
|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Circle  | Surface Area $A = \pi R^2$ <br>Circumference $u = 2 \pi R$ <br>Diameter $d = 2 R$                                                                                                                                                        |
| Sphere  | Volume $V = \frac{4}{3} \pi R^3$ <br>Surface Area $A = 4 \pi R^2$                                                                                                                                                                    |
| Ellipse | Approximate everything by using formulas for a circle lol <br>The semi-major axis is called $\, a$: <br><img style="width: 250px" src="/images/astrophysics/ellipse.png"/><br>The orbits of planets are ellipses, but can often be approximated as circles |




Gravitational force: $F_G = G \frac{m_E M_\odot}{r_E^2}$

Calculate distance from arcseconds (Speedrun strategy): $\frac{1}{X''} = Y \, \text{pc}$

Velocity and mass for a circular orbit: $v = \sqrt{\frac{G M}{r}} \Rightarrow M = v^2 \frac{r}{G}$

Orbital period for a circular orbit: $T = \frac{2 \pi r}{v}$





Energy produced by a star per second in Joule: $\Delta E = \mathcal{L_\*} \cdot 1 \, \text{s} \; \text{(J)}$ 



Virial theorem: $2 \cdot E_\text{thermal} + E_\text{grav} = 0$

Life time of a star in years: $T_\* = T_\odot \cdot \( \frac{M_\*}{M_\odot} \)^{-2.5} \, \text{(years)}$



Flux of a star: $s_\* = \( \frac{\mathcal{L_\*}}{4 \pi a_P^2} \)$

Albedo (fraction of the energy coming from sunlight that a planet reflects) = 1 - X (where X is the fraction of the energy of the sun that the planet absorbs)

Equilibrium temperature of a planet: $T_P = ( \frac{X \cdot s_\*}{4 \cdot \sigma_{SB}} )^{0.25}\; \text{(K)}$



Absolute Magnitude (the brighter a star, the smaller its magnitude): $M = m - 5 \cdot \log_{10} ( \frac{d}{10 \, \text{pc}} )$  where $m$ is the relative magnitude and $d$ the distance from the observer (us) to the star






- https://de.wikipedia.org/wiki/Zweik%C3%B6rperproblem
- https://www.theexpertta.com/book-files/OpenStaxAstronomy/Astronomy_20.5.%C2%A0The%20Life%20Cycle%20of%20Cosmic%20Material_pg714%20-%20716.pdf
- https://en.wikipedia.org/wiki/Milky_Way


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