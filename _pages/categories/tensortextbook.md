---
layout: archive
permalink: /tensortextbook/
title: "딥러닝 텐서플로 교과서"

author_profile: true
sidebar:
  nav: "docs"
---

{% assign posts = site.categories.tensortextbook %}
{% for post in posts %}
  {% include custom-archive-single.html type=entries_layout %}
{% endfor %}