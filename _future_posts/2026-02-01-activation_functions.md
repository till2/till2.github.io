---
layout: post
title:  "Activation Functions that enable Multiplication"
author: "Till Zemann"
date:   2026-02-01 01:09:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
tags: [machine learning, optimization]
thumbnail: "/images/about/ghpb1.png"
---


<div class="img-block" style="width: 300px;">
    <img src="/images/about/ghpb1.png"/>
</div>

<h3> Contents </h3>
* TOC
{:toc}

### Introduction

Hi!

The SwiGLU activation function is a variant of the Gated Linear Unit (GLU). Unlike a standard activation function that operates on a single input, SwiGLU involves two linear projections of the input $x$, typically denoted as $W$ and $V$.

Formally, SwiGLU is defined as the component-wise product of a "value" path and a "gate" path, where the gate is activated by the Swish function:

$$
\text{SwiGLU}(x, W, V, \beta) = \text{Swish}_{\beta}(xW) \otimes (xV)
$$

Expanding the Swish term ($\text{Swish}_{\beta}(z) = z \cdot \sigma(\beta z)$), the explicit equation becomes:

$$
\text{SwiGLU}(x) = \underbrace{(xW \cdot \sigma(\beta (xW)))}_{\text{Swish Gate}} \cdot \underbrace{(xV)}_{\text{Linear Value}}
$$

Where:
*   $x$: The input vector.
*   $W$: The weight matrix for the **gate** mechanism.
*   $V$: The weight matrix for the **value** pass-through.
*   $\beta$: The learnable or fixed parameter controlling the Swish steepness.
*   $\otimes$: Component-wise multiplication.

<!-- Interactive Plot -->
<script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
<div id="swiglu-plot" style="width:100%; max-width:700px; height:500px; margin: 0 auto;"></div>

<div style="width:100%; max-width:700px; margin: 20px auto; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background: #f9f9f9;">
    <p style="margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 5px;"><strong>SwiGLU Parameters:</strong></p>
    
    <!-- Beta Slider -->
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <label for="beta-slider" style="width: 150px; font-family: monospace;">Beta ($\beta$): <span id="beta-val">1.0</span></label>
        <input type="range" id="beta-slider" min="0.1" max="5" step="0.1" value="1.0" style="flex-grow: 1;">
    </div>

    <!-- Gate Weight Slider -->
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <label for="w-slider" style="width: 150px; font-family: monospace;">Gate Weight ($W$): <span id="w-val">1.0</span></label>
        <input type="range" id="w-slider" min="-2" max="2" step="0.1" value="1.0" style="flex-grow: 1;">
    </div>

    <!-- Value Weight Slider -->
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <label for="v-slider" style="width: 150px; font-family: monospace;">Value Weight ($V$): <span id="v-val">1.0</span></label>
        <input type="range" id="v-slider" min="-2" max="2" step="0.1" value="1.0" style="flex-grow: 1;">
    </div>
</div>

<script>
    function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
    
    // SwiGLU = Swish(xW) * (xV)
    // Swish(z) = z * sigmoid(beta * z)
    function swiglu(x, beta, w, v) {
        let gate_input = x * w;
        let swish_gate = gate_input * sigmoid(beta * gate_input);
        let value_path = x * v;
        return swish_gate * value_path;
    }

    function generateData(beta, w, v) {
        let xValues = [];
        let yValues = [];
        // Plot from -3 to 3 to see the behavior clearly around 0
        for (let i = -3; i <= 3; i += 0.05) {
            let x = Math.round(i * 100) / 100;
            xValues.push(x);
            yValues.push(swiglu(x, beta, w, v));
        }
        return { x: xValues, y: yValues };
    }

    // Initial State
    let state = { beta: 1.0, w: 1.0, v: 1.0 };
    let data = generateData(state.beta, state.w, state.v);

    let layout = {
        title: 'SwiGLU: Swish(xW) · (xV)',
        xaxis: { title: 'Input (x)', range: [-3, 3] },
        yaxis: { title: 'Output (y)', range: [-2, 5] },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 50, r: 20, l: 50, b: 50 }
    };

    let config = { responsive: true, displayModeBar: false };
    
    Plotly.newPlot('swiglu-plot', [{
        x: data.x, 
        y: data.y, 
        mode: 'lines', 
        line: { color: '#1E72E7', width: 3 }, 
        name: 'SwiGLU'
    }], layout, config);

    function updatePlot() {
        // Read all values
        state.beta = parseFloat(document.getElementById('beta-slider').value);
        state.w = parseFloat(document.getElementById('w-slider').value);
        state.v = parseFloat(document.getElementById('v-slider').value);

        // Update Labels
        document.getElementById('beta-val').innerText = state.beta.toFixed(1);
        document.getElementById('w-val').innerText = state.w.toFixed(1);
        document.getElementById('v-val').innerText = state.v.toFixed(1);

        // Regenerate and plot
        let newData = generateData(state.beta, state.w, state.v);
        Plotly.restyle('swiglu-plot', { y: [newData.y] });
    }

    // Attach listeners
    document.getElementById('beta-slider').addEventListener('input', updatePlot);
    document.getElementById('w-slider').addEventListener('input', updatePlot);
    document.getElementById('v-slider').addEventListener('input', updatePlot);
</script>


<p class="vspace"></p>

### References
1. [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)

{% if page.comments %}
<p class="vspace"></p>
<a class="commentlink" role="button" href="/comments/">Post a comment.</a>
{% endif %}

{% if page.back_to_top_button %}
<script src="https://unpkg.com/vanilla-back-to-top@7.2.1/dist/vanilla-back-to-top.min.js"></script>
<script>addBackToTop({
  diameter: 40,
  backgroundColor: 'rgb(255, 255, 255, 0.7)',
  textColor: '#4a4946'
})</script>
{% endif %}