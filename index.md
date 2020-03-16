---
title: Deep Neural Networks Module
---

You can find the [Syllabus here](./syllabus.html) and the [Assessment here](./assessment.html).

---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---


## People

{% assign lastone = site.lectures | last %}
{% for person in site.lectures %}
{% include listlectures.html %}
{% endfor %}

