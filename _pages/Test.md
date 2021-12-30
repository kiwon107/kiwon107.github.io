---
layout: archive
permalink: test
title: "Test"

author_profile: true
sidebar:
  nav: "docs"
---

{% assign posts = site.categories.test %}
{% for post in posts %}
  {% include custom-archive-single.html type=entries_layout %}
{% endfor %}