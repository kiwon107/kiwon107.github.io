---
layout: archive
permalink: /elecbasic/
title: "전기 기초"

author_profile: true
sidebar:
  nav: "docs"
---

{% assign posts = site.categories.elecbasic %}
{% for post in posts %}
  {% include custom-archive-single.html type=entries_layout %}
{% endfor %}