---
layout: archive
permalink: /paper/
title: "논문"

author_profile: true
sidebar:
  nav: "docs"
---

{% assign posts = site.categories.paper %}
{% for post in posts %}
  {% include custom-archive-single.html type=entries_layout %}
{% endfor %}