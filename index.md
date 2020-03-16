---
layout: home
---


{% assign lastone = site.lectures | last %}
{% for lecture in site.lectures %}
{% include listlecture.html %}
{% endfor %}

## Special Topics

{% assign lastone = site.special_topics | last %}
{% for topic in site.special_topics %}
{% include listtopic.html %}
{% endfor %}
