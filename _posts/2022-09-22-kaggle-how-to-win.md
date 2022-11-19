---
layout: post
title:  "Kaggle"
author: "Till Zemann"
date:   2022-09-20 20:36:41 +0200
categories: jekyll update
comments: true
back_to_top_button: true
math: true
---

## Introduction
[Kaggle](https://www.kaggle.com/) is an online platform that hosts Machine Learning competitions.
The Kaggle [ranks](https://www.kaggle.com/rankings) are distributed according to:

<div class="table-wrap">
    <table class="prob-table">
        <tr>
            <td>             <strong> Rank     </strong> </td>
            <td colspan="3"> <strong> #Holders     </strong> </td>
        </tr>
        <tr>
            <td> Grandmaster </td>
            <td> 262 </td>
        </tr>
        <tr>
            <td> Master </td>
            <td> 1,843 </td>
        </tr>
        <tr>
            <td> Expert </td>
            <td> 8,191 </td>
        </tr>
        <tr>
            <td> Contributer </td>
            <td> 70,159 </td>
        </tr>
        <tr>
            <td> Novice </td>
            <td> 102,352 </td>
        </tr>
    </table>
</div>


## Build a good ML model in 3 stages
1. Turn your business problem into a ML problem (build the dataset right)
2. Build a good ML model 
    - pick the right approach
    - do good feature engineering
    - statistically evaluate the model using [(k-fold) Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
    - use a regularizer or dropout to avoid overfitting
5. Productionize the model

## High performing models
1. Gradient boosting ([XGBoost](https://xgboost.readthedocs.io/en/stable/)) seems to outperform random forests (also an ensemble method) by a little bit
2. Random forests (the ensemble variant of decision trees)

## What seperates winning entries from others
- Good __feature engineering__. Here are some creative examples from top submissions:
    - Extend a text dataset with Googe-Translate (non-linear, because if you translate to another language and then back again the result won't necessarily be the original sentence) from $A \rightarrow B \rightarrow A'$
    with $A$ and $B$ being the same sentence in different languages.
    - Extract a lot (>70) of different features from just a date or timestamp, such as the season (summer/winter), if it's weekday or weekend, also merging with other data such as events (holiday or not?)
- Appropriate and rich __image augmentation__ for computer vision (CV) tasks
- For Reinforcement Learning competitions with a simulator that only offers a couple of testruns a day, an excellent strategy is to actually rebuild the simulator (mimick the rewards and structure as close as possible within reasonable timespans) to have a local evaluation tool available for tuning


### Reinforcement Learning in Kaggle
Kaggle now also invests into Reinforcement Learning through simulation-based competitions.

## References
A lot of info from this blogpost comes an [interview with Anthony Goldbloom][anthony-goldbloom-how-to-win-kaggle-competitions], founder of Kaggle. He talks about approaches that are commonly used by winning competetors and how submissions evolved over the years as Data Science matured and Deep Learning entered the field as another competitive approach. 

<!-- Normal Text and Highlights -->
<!-- Text with Colors -->
<!-- Math Text -->
<!-- Tables -->
<!-- Code Box -->
<!-- In-Text Citing -->
<!-- Images -->

<!-- References -->
[anthony-goldbloom-how-to-win-kaggle-competitions]: https://www.youtube.com/watch?v=0ZJQ2Vsgwf0
[aladdin-persson-top-1-percent-no-ensemble]: https://www.youtube.com/watch?v=MOnk75_8b9M
[how-to-win-kaggle-competition-master-advice]: https://www.youtube.com/watch?v=in0miFAiwZA


<!-- Optional Comment Section-->
{% if page.comments %}
<p class="vspace"></p>
<a class="commentlink" role="button" href="/comments/">Post a comment.</a> <!-- role="button"  -->
{% endif %}

<!-- Optional Back to Top Button -->
{% if page.back_to_top_button %}
<script src="https://unpkg.com/vanilla-back-to-top@7.2.1/dist/vanilla-back-to-top.min.js"></script>
<script>addBackToTop({
  diameter: 45,
  backgroundColor: 'rgb(30,144,255, 0.7)',
  textColor: '#fff'
})</script>
{% endif %}