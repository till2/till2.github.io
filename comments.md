---
layout: page
title: Comments 💬
comments: true
permalink: /comments/
---


### <em> This is the place for suggestions, questions, exchange of ideas and more. </em>

<div class="img-block" style="width: 300px;">
    <img src="/images/cozy-wall.png"/>
</div>

<br>

<!-- Upgrade comment system to embedded GitHub comments: 
	https://aristath.github.io/blog/static-site-comments-using-github-issues-api 
-->

<!-- Comment section-->
{% if page.comments %}
<div id="disqus_thread"></div>
<script>

    (function() { // DON'T EDIT BELOW THIS LINE
    var d = document, s = d.createElement('script');
    s.src = 'https://till2-github-io.disqus.com/embed.js';
    s.setAttribute('data-timestamp', +new Date());
    (d.head || d.body).appendChild(s);
    })();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>
{% endif %}