---
layout: archive
permalink: /prob/
title: "확률과 통계"

author_profile: true
sidebar:
  nav: "docs"
---

{% assign posts = site.categories.prob %}
{% for post in posts %}
  {% include custom-archive-single.html type=entries_layout %}
{% endfor %}