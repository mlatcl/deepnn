---
layout: home
title: Deep Neural Networks
---

{% for lecture in site.lectures %}
{% include listlecture.html %}
{% endfor %}

## Special Topics

The lectures for weeks 5 to 8 will focus on "special topics". Some of these will be given by guest lecturers and therefore availability will be dependent on availability of guest lecturers.

{% for topic in site.special_topics %}
{% include listtopic.html %}
{% endfor %}
