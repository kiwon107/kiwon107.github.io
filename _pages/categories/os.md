---
layout: archive
permalink: /os/
title: "운영체제"

author_profile: true
sidebar:
  nav: "docs"
---

{% assign posts = site.categories.os %}
{% for post in posts %}
  {% include custom-archive-single.html type=entries_layout %}
{% endfor %}